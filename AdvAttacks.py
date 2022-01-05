import random
import os
import math

import numpy as np
# from attacks import Attack

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.models as tormodel



'''-----------------------------PGD Attacks--------------------------'''
### PGD or FSGM(config['num_steps']=1)
def attack_PGD(model, x_natural, class_label, config, train_=False):
    '''
    config['rand_start']
    config['epsilon']
    config['num_steps']
    config['step_size']
    '''
    # define certiria and model
    x_natural, class_label = x_natural.cuda(), class_label.cuda()
    model.eval()    # turn on evaluation model to generate the attack samples
    x = x_natural.detach()
    if train_ is True and config['rand_start'] is True:
        x = x + torch.zeros_like(x).uniform_(-config['epsilon'], config['epsilon'])
    for i in range(config['num_steps']):
        x.requires_grad_()
        with torch.enable_grad():
            _, _, output = model(x)
            loss = F.cross_entropy(output, class_label, size_average=False)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + config['step_size']*torch.sign(grad.detach())
        x = torch.min(torch.max(x, x_natural - config['epsilon']), x_natural + config['epsilon'])
        x = torch.clamp(x, 0.0, 1.0)
    img_adv = x.cuda()
    model.train() # turn on train model to do the domain and label calssification
    return img_adv



'''-----------------------------FAB Attacks--------------------------'''
from fab import FAB
def attack_fab(model, x_natural, class_label, config):
    '''
    config['fab_num_steps'] = 100
    config['fab_epsilon'] = None
    config['fab_n_restarts'] = 1
    config['fab_alpha_max'] = 0.1
    config['fab_eta']=1.05
    config['fab_beta']=0.9
    config['fab_loss_fn']=None
    config['fab_verbose']=False
    config['fab_seed'] = 0
    config['fab_targeted'] = False
    config['fab_n_classes'] = 10
    '''
    attack = FAB(model, norm='Linf', 
                steps=config['fab_num_steps'], 
                eps=config['fab_epsilon'], 
                n_restarts=config['fab_n_restarts'], 
                alpha_max=['fab_alpha_max'], 
                eta=['fab_eta'], 
                beta=['fab_beta'], 
                loss_fn=None, verbose=False, seed=0, 
                targeted=False, n_classes=10)
    adv_images = attack(x_natural, class_label)
    return adv_images



'''-----------------------------T-PGD--------------------------'''
### TRADES, KL-PGD
def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def Loss_KLD(outputs, targets, args):
    T=args['t']
    log_softmax_outputs = F.log_softmax(outputs/T, dim=1)
    softmax_targets = F.softmax(targets/T, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def attack_TRADES(model, x_natural, config):
    '''
    config['distance'] = 'l_inf' or 'l_2'
    config['num_steps'] = attack steps
    config['step_size'] = single step size
    config['epsilon'] = attack radii
    '''
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    x_natural = x_natural.cuda()
    model.eval()    # turn on evaluation model to generate the attack samples
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if config['distance'] == 'l_inf':
        for _ in range(config['num_steps']):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, _, output_adv = model(x_adv)
                _, _, output_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(output_adv, dim=1),
                                       F.softmax(output_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + config['step_size'] * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - config['epsilon']), x_natural + config['epsilon'])
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif config['distance'] == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=config['epsilon'] / config['step_size'] * 2)

        for _ in range(config['step_size']):
            x_adv = x_natural + delta
            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                _, _, output_adv = model(x_adv)
                _, _, output_nat = model(x_natural)
                loss = (-1) * criterion_kl(F.log_softmax(output_adv, dim=1),
                                           F.softmax(output_nat, dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()
            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=config['epsilon'])
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train() # turn on train model to do the domain and label calssification
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv



'''-----------------------------L_inf C&W Attacks--------------------------'''
### C&W
DECREASE_FACTOR = 0.9   # 0<f<1, rate at which we shrink tau; larger is more accurate
MAX_ITERATIONS = 100   # number of iterations to perform gradient descent
ABORT_EARLY = True      # abort gradient descent upon first valid solution
INITIAL_CONST = 1e-5    # the first value of c to start at
LEARNING_RATE = 5e-3    # larger values converge faster to less accurate results
LARGEST_CONST = 2e+1    # the largest value of c to go up to before giving up
REDUCE_CONST = False    # try to lower c each iteration; faster to set to false
TARGETED = True        # should we target one specific class? or just be wrong?
CONST_FACTOR = 2.0      # f>1, rate at which we increase constant, smaller better
NUM_CLASSES = 10

def _f(target_cls, adv_imgs, labels, args_):
        outputs = target_cls(adv_imgs)
        y_onehot = torch.nn.functional.one_hot(labels)

        real = (y_onehot * outputs).sum(dim=1)
        other, _ = torch.max((1-y_onehot)*outputs, dim=1)

        if args_['cw_targeted']:
            loss = torch.clamp(other-real, min=-args_['cw_kappa'])
        else:
            loss = torch.clamp(real-other, min=-args_['cw_kappa'])
        return loss

def arctanh(imgs):
    scaling = torch.clamp(imgs, max=1, min=-1)
    x = 0.999999 * scaling
    return 0.5*torch.log((1+x)/(1-x))

def scaler(x_atanh):
    return ((torch.tanh(x_atanh))+1) * 0.5

def attack_CW(model, x_natural, class_label, config):
    '''
    config['cw_targeted']
    config['cw_c']
    config['cw_kappa']
    config['cw_n_iters']
    config['cw_lr']
    config['cw_binary_search_steps']
    '''
    x_natural, class_label = x_natural.cuda(), class_label.cuda()
    model.eval()    # turn on evaluation model to generate the attack samples
    x_arctanh = x_natural.detach()
    x_arctanh = arctanh(x_arctanh)

    for _ in range(config['cw_binary_search_steps']):
        delta = torch.zeros_like(x_arctanh).cuda()
        delta.detach_()
        delta.requires_grad = True
        optimizer = torch.optim.Adam([delta], lr=config['cw_lr'])
        prev_loss = 1e6

        for step in range(config['cw_n_iters']):
            optimizer.zero_grad()
            adv_examples = scaler(x_arctanh + delta)
            loss1 = torch.sum(config['cw_c']*_f(model, adv_examples, class_label, config))
            loss2 = F.mse_loss(adv_examples, x_natural, reduction='sum')
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            if step % (config['cw_n_iters'] // 10) == 0:
                if loss > prev_loss:
                    break
                prev_loss = loss
    adv_imgs = scaler(x_arctanh + delta).detach()
    return adv_imgs

    

'''-----------------------------Teacher-Student-PGD--------------------------'''
### Teacher-Student KL-PGD
def attack_KLPGD(teacher, student, x_natural, config):
    # define certiria and student
#     criterion_ce = nn.cross_entropy(size_average=False)
    student.eval()    # turn on evaluation student to generate the attack samples
    teacher.eval()
    # 
    x = x_natural.detach()
    if config['rand_start'] == 1:
        x = x + torch.zeros_like(x).uniform_(-config['epsilon'], config['epsilon'])
    for i in range(config['num_steps']):
        x.requires_grad_()
        with torch.enable_grad():
            loss = Loss_KLD(student(x), teacher(x), T=1.0)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + config['step_size']*torch.sign(grad.detach())
        x = torch.min(torch.max(x, x_natural - config['epsilon']), x_natural + config['epsilon'])
        x = torch.clamp(x, 0.0, 1.0)
    img_adv = x.cuda()
    student.train() # turn on train student to do the domain and label calssification
    return img_adv

