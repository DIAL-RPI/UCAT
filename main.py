import random
import os
import math

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import imgaug.augmenters as iaa
import imgaug as ia
from PIL import Image

from sklearn import manifold, datasets

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import foolbox as fb
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


from torchvision import datasets, transforms
from torchvision.utils import make_grid

import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import time
# import ipdb
from IPython.core.debugger import Pdb
ipdb = Pdb()
import gc

from models.model_construct import Model_Construct # for the model construction
from trainer_ucat_PL import train # for the training process
from trainer_ucat_PL import validate, validate_compute_cen # for the validation/test process
from trainer_ucat_PL import k_means, spherical_k_means, kernel_k_means # for K-means clustering and its variants
from trainer_ucat_PL import source_select # for source sample selection
from PseudoLabeling import PseudoLabeling

from data.prepare_data import generate_dataloader # prepare the data and dataloader
from utils.consensus_loss import ConsensusLoss

from AdvAttacks import attack_PGD, attack_TRADES

os.environ["CUDA_VISIBLE_DEVICES"] = "put_your_gpu_#"

def mkdir_if_missing(save_dir):
    if os.path.exists(save_dir):
        return 1
    else:
        os.makedirs(save_dir)
        return 0

def save_model(net, model_root, filename):
    """Save trained model."""
    flag = mkdir_if_missing(model_root)
    torch.save(net.module.state_dict(), os.path.join(model_root, filename))

def count_epoch_on_large_dataset(train_loader_target, train_loader_source, args):
    batch_number_t = len(train_loader_target)
    batch_number = batch_number_t
    if args['src_cls']:
        batch_number_s = len(train_loader_source)
        if batch_number_s > batch_number_t:
            batch_number = batch_number_s
    return batch_number
    

def save_checkpoint(state, is_best, args):
    filename = 'checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))
        
def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        layer.weight.data.normal_(0, math.sqrt(2. / n))
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
    elif layer_name.find("Linear") != -1:
        layer.bias.data.zero_()

def init_model(net, restore=None, device=True):
    """Init models with cuda and weights."""
    if device==True:
        cudnn.benchmark = True
        net.cuda()
        net = nn.DataParallel(net) 
    ## init weights of model
#     net.apply(init_weights)
    
    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.module.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))
    return net


# random seeds
cudnn.benchmark = True
manual_seed = 41
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)

####################### Settings #################################
args = {}
# datasets
args['data_path_source'] = '/your_dataset/Office31/' # split with ratio = 4:1
args['data_path_target'] = '/your_dataset/Office31/'
args['data_path_target_t'] = '/your_dataset/Office31/'
args['src'] = 'amazon' # 'A' 
args['tar'] = 'webcam_80' # 'W' for train, 80%
args['tar_t'] = 'webcam_20' # 'W' for test, 20%
args['no_da'] = True
args['num_classes'] = 31
args['batch_size'] = 93

# source sample selection
args['src_soft_select'] = False
args['src_hard_select'] = False
args['src_mix_weight'] = False
args['tao_param'] = 0.5


# general optimization options
args['epochs'] = 250
args['switch_epoch'] = 100
args['workers'] = 8
args['lr'] = 1e-2 #lr = 0.01 for orginal ardc
args['no_da'] = False # No data augmentation
args['lr_plan'] = 'dao'
args['schedule'] = [60, 120, 200]
args['momentum'] = 0.9
args['weight_decay'] = 1e-4
args['nesterov'] = False
args['eps'] = 1e-6


# specific optimization options
args['ao'] = False
args['cluster_method'] = 'kmeans'
args['cluster_iter'] = 5
args['cluster_kernel'] ='rbf'
args['gamma'] = None
args['sample_weight'] = False
args['initial_cluster'] = 1 # 'target or source class centroids for initialization of K-means'
args['init_cen_on_st'] = False
args['src_cen_first'] = False
args['src_cls'] = False
args['src_fit'] = False
args['src_pretr_first'] = False
args['learn_embed'] = False
args['no_second_embed'] = False
args['alpha'] = 1.0
args['beta'] = 1.0
args['embed_softmax'] = False
args['div'] = 'kl'
args['gray_tar_agree'] = False
args['aug_tar_agree'] = False
args['sigma'] = 0.1
     
     
# checkpoints
args['resume'] = ''
# args['log'] = './log_dir' # if needed
args['stop_epoch'] = 250 # if needed
args['iSave'] = True
args['save_dir'] = '/your_model_save_location'
mkdir_if_missing(args['save_dir'])

# architecture
args['arch'] = 'resnet50'
args['num_neurons'] = 128
args['pretrained'] = False
args['print_freq'] = 10

# args[''] = 
args['pretrained'] = True
if args['tar'].find('amazon') == -1:
    args['init_cen_on_st'] = True
elif args['src'].find('webcam') != -1:
    args['beta'] = 0.5
args['src_cls'] = True
args['src_cen_first'] = True
args['learn_embed'] = True
args['embed_softmax'] = True

# attacks
args['rand_start'] = False
args['epsilon'] = 0.031
args['step_size'] = 0.007
args['num_steps'] = 10
args['distance'] = 'l_inf'

# process data and prepare dataloaders
train_loader_source, train_loader_target, val_loader_target, val_loader_target_t, val_loader_source = generate_dataloader(args)
train_loader_target.dataset.tgts = list(np.array(torch.LongTensor(train_loader_target.dataset.tgts).fill_(-1)))




############## train ###################
best_prec1 = 0
best_test_prec1 = 0
cond_best_test_prec1 = 0
best_cluster_acc = 0 
best_cluster_acc_2 = 0

# global args, best_prec1, best_test_prec1, cond_best_test_prec1, best_cluster_acc, best_cluster_acc_2
    
# define model
model = Model_Construct(args)
model = torch.nn.DataParallel(model).cuda() # define multiple GPUs
    
# define learnable cluster centers
learn_cen = Variable(torch.cuda.FloatTensor(args['num_classes'], 2048).fill_(0))
learn_cen.requires_grad_(True)
learn_cen_2 = Variable(torch.cuda.FloatTensor(args['num_classes'], args['num_neurons'] * 4).fill_(0))
learn_cen_2.requires_grad_(True)

# define loss function/criterion and optimizer
criterion = torch.nn.CrossEntropyLoss().cuda()
criterion_cons = ConsensusLoss(nClass=args['num_classes'], div=args['div']).cuda()


# apply different learning rates to different layer
optimizer = torch.optim.SGD([
                            {'params': model.module.conv1.parameters(), 'name': 'conv'},
                            {'params': model.module.bn1.parameters(), 'name': 'conv'},
                            {'params': model.module.layer1.parameters(), 'name': 'conv'},
                            {'params': model.module.layer2.parameters(), 'name': 'conv'},
                            {'params': model.module.layer3.parameters(), 'name': 'conv'},
                            {'params': model.module.layer4.parameters(), 'name': 'conv'},
                            {'params': model.module.fc1.parameters(), 'name': 'ca_cl'},
                            {'params': model.module.fc2.parameters(), 'name': 'ca_cl'},
                            {'params': learn_cen, 'name': 'conv'},
                            {'params': learn_cen_2, 'name': 'conv'}],
                            lr=args['lr'],
                            momentum=args['momentum'],
                            weight_decay=args['weight_decay'], 
                            nesterov=args['nesterov'])


### resume pretrained source model
epoch = 0
init_state_dict = model.state_dict()
if args['resume']:
    if os.path.isfile(args['resume']):
        print("==> loading checkpoints '{}'".format(args['resume']))
        checkpoint = torch.load(args['resume'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError('The file to be resumed from does not exist!', args['resume'])

print('begin training')
batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source, args) # max(src_len, tgt_len)
num_itern_total = args['epochs'] * batch_number # total iteration number

new_epoch_flag = False # if new epoch, new_epoch_flag=True
test_flag = False # if test, test_flag=True

src_cs = torch.cuda.FloatTensor(len(train_loader_source.dataset.tgts)).fill_(1) # initialize source weights


### target label generator
srdc_restore = os.path.join('your_label_generator.pth')
lab_gen = init_model(net=Model_Construct(args), restore=srdc_restore)

### PL generated by PseudoLabeling(model, class_num, interval_num, dataloader)
Pth = PseudoLabeling(model=lab_gen, class_num=31, interval_num=100, dataloader=val_loader_source)
### our calculated Pth is given below:
# Pth = torch.Tensor([0.4900, 0.9700, 0.3100, 0.9200, 0.8200, 0.8400, 0.9100, 0.9000, 0.6700,
#                     0.9000, 0.7700, 0.9100, 0.9100, 0.3700, 0.7700, 0.9400, 0.9500, 0.8800,
#                     0.6300, 0.9100, 0.5900, 0.6400, 0.9700, 0.3000, 0.8200, 0.7900, 0.7900,
#                     0.4000, 0.7100, 0.5900, 0.4900])


count_itern_each_epoch = 0
epoch = 0
for itern in range(epoch * batch_number, num_itern_total):
    # evaluate on the target training and test data
    if (itern == 0) or (count_itern_each_epoch == batch_number):
        prec1, c_s, c_s_2, c_t, c_t_2, c_srctar, c_srctar_2, source_features, source_features_2, source_targets, target_features, target_features_2, target_targets, pseudo_labels = validate_compute_cen(val_loader_target, val_loader_source, model, criterion, epoch, args, keys=None)
        
        print('============================================================')
        test_acc = validate(val_loader_target_t, model, criterion, epoch, args)
        test_acc = validate(val_loader_target_t, model, criterion, epoch, args, True)
        print('============================================================')
        test_flag = True
        
        # K-means clustering or its variants
        if ((itern == 0) and args['src_cen_first']) or (args['initial_cluster'] == 2):
            cen = c_s
            cen_2 = c_s_2
        else:
            cen = c_t
            cen_2 = c_t_2
        
        if (itern != 0) and (args['initial_cluster'] != 0) and (args['cluster_method'] == 'kernel_kmeans'):
            cluster_acc, c_t = kernel_k_means(target_features, target_targets, pseudo_labels, train_loader_target, 
                                              epoch, model, args, best_cluster_acc)
            cluster_acc_2, c_t_2 = kernel_k_means(target_features_2, target_targets, pseudo_labels, train_loader_target, 
                                                  epoch, model, args, best_cluster_acc_2, change_target=False)
        elif args['cluster_method'] != 'spherical_kmeans':
            cluster_acc, c_t = k_means(target_features, target_targets, train_loader_target, 
                                       epoch, model, cen, args, best_cluster_acc)
            cluster_acc_2, c_t_2 = k_means(target_features_2, target_targets, train_loader_target, 
                                           epoch, model, cen_2, args, best_cluster_acc_2, change_target=False)
        elif args['cluster_method'] == 'spherical_kmeans':
            cluster_acc, c_t = spherical_k_means(target_features, target_targets, train_loader_target, 
                                                 epoch, model, cen, args, best_cluster_acc)
            cluster_acc_2, c_t_2 = spherical_k_means(target_features_2, target_targets, train_loader_target, 
                                                     epoch, model, cen_2, args, best_cluster_acc_2, change_target=False)
        print('cluster_acc : {}, cluster_acc_2 : {}'.format(cluster_acc, cluster_acc_2))
            
        # re-initialize learnable cluster centers
        if args['init_cen_on_st']:
            cen = (c_t + c_s) / 2 # or c_srctar
            cen_2 = (c_t_2 + c_s_2) / 2 # or c_srctar_2
        else:
            cen = c_t
            cen_2 = c_t_2
        #if itern == 0:
        learn_cen.data = cen.data.clone()
        learn_cen_2.data = cen_2.data.clone()
            
        # select source samples
        if (itern != 0) and (args['src_soft_select'] or args['src_hard_select']):
            src_cs = source_select(source_features, source_targets, target_features, 
                                   pseudo_labels, train_loader_source, 
                                   epoch, c_t.data.clone(), args)
        
        # use source pre-trained model to extract features for first clustering
        if (itern == 0) and args['src_pretr_first']: 
            model.load_state_dict(init_state_dict)
                
        if itern != 0:
            count_itern_each_epoch = 0
            epoch += 1
        batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source, args)
        train_loader_target_batch = enumerate(train_loader_target)
        train_loader_source_batch = enumerate(train_loader_source)
        
        new_epoch_flag = True
        del source_features
        del source_features_2
        del source_targets
        del target_features
        del target_features_2
        del target_targets
        del pseudo_labels
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    
    # record the best prec1 and save checkpoint    
    if test_flag:
        if prec1 > best_prec1:
            best_prec1 = prec1
            cond_best_test_prec1 = 0
        if test_acc > best_test_prec1:
            best_test_prec1 = test_acc
#         ipdb.set_trace()
        is_cond_best = ((prec1 == best_prec1) and (test_acc > cond_best_test_prec1))
        if is_cond_best:
            cond_best_test_prec1 = test_acc
        print('best_prec1:{}, best_test_prec1:{}, cond_best_test_prec1:{}'.format(
                            best_prec1, best_test_prec1, cond_best_test_prec1))
        test_flag = False
    
    ### save models
    if args['iSave']==True:
        save_model(net=model, model_root=args['save_dir'], filename='srdc_'+args['arch']+'_epoch'+str(epoch))

    ### switching target label generator and re-calibrate Pth   
    if epoch == args['switch_epoch']:
        srdc_restore = os.path.join('your_tgt_label_generator.pth')
        lab_gen = init_model(net=Model_Construct(args), restore=srdc_restore)
        ### PL generated by PseudoLabeling(model, class_num, interval_num, dataloader)
        Pth = PseudoLabeling(model=lab_gen, class_num=31, interval_num=100, dataloader=val_loader_source)
    
    # early stop
    if epoch > args['stop_epoch']:
        break

    # train for one iteration
    train_loader_source_batch, train_loader_target_batch = train(train_loader_source, train_loader_source_batch, 
                                                                 train_loader_target, train_loader_target_batch, model, 
                                                                 learn_cen, learn_cen_2, criterion_cons, optimizer, itern, epoch, 
                                                                 new_epoch_flag, src_cs, 
                                                                 args, Th_classwise=Pth, keys='T', lab_generator=lab_gen)

    model = model.cuda()
    new_epoch_flag = False
    count_itern_each_epoch += 1
