import random
import os
import math
import numpy as np
import matplotlib.pyplot as plt

### K-way classification problem
### Divide the confidence range [0,1] into Mintervals
def PseudoLabeling(model, class_num, interval_num, dataloader):
    M = interval_num
    K = class_num
    model = model.cuda()

    inte_m = 1.0/M            # interval
    delta_m = np.zeros((K,M)) # counts
    conf_m = np.zeros((K,M))  # confidence
    acc_m = np.zeros((K,M))   # accuracy

    data_tgt_iter = iter(dataloader)
    test_len = len(data_tgt_iter)
    # print(test_len)

    ### compute elements
    ii = 0
    while ii<test_len:
        ii+=1
        input_img_tgt, lab_tgt = next(data_tgt_iter)
        input_img_tgt, lab_tgt = input_img_tgt.cuda(), lab_tgt.cuda()
        class_output = model(input_img_tgt)
        prob = nn.functional.softmax(class_output)
        pred = class_output.data.max(1, keepdim=True)[1]

        prob_np = prob.data.cpu().numpy() # prob
        pred_np = pred.data.cpu().numpy() # y_hat
        label_np = lab_tgt.cpu().numpy()  # y_target

        for ind in np.arange(len(label_np)):
            kk = np.where(prob_np[ind]==max(prob_np[ind]))[0]
            prob_c = prob_np[ind][kk]
            mm = np.int((prob_c)/inte_m)-1

            delta_m[kk, mm]+=1
            conf_m[kk,mm]+=prob_c
            if kk == label_np[ind]: 
                acc_m[kk,mm]+=1

    ### compute
    Pth = np.zeros(K)
    for kk in np.arange(K):
        I_conf=conf_m[kk,:]/(delta_m[kk,:]+1e-9) # I_confidence of class ind
        I_acc=acc_m[kk,:]/(delta_m[kk,:]+1e-9) # I_accuracy of class ind
        X_m=delta_m[kk] # number of counts of class ind

        M_ = len(X_m)
        tao = np.zeros(M_)
        X_mod = sum(X_m) # total numer of samples 

        for ind in np.arange(M_):
            X_mod = X_mod - X_m[ind] # total numer of samples within current interval
            tao[ind] = 0
            count = 0
            for mm in np.arange(ind,M_):
                if X_m[mm]>0:
                    tao[ind] +=  I_acc[mm] * (1-abs(I_conf[mm]-I_acc[mm])) #X_m[mm]/(X_mod+1e-6) * #1/(M_-ind)*
                    count += 1
            tao[ind] = tao[ind]/(count+1e-6)*X_m[ind:].sum()/(X_m.sum()+1e-6)
        Pth[kk] = max(np.where(tao==tao.max())[0]-1) * inte_m
    return Pth