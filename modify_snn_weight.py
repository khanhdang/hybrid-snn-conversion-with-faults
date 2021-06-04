#---------------------------------------------------
# Imports
#---------------------------------------------------
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torchviz import make_dot
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import datetime
import pdb
from self_models import *
import sys
import os
import shutil
import argparse

error_type = 'stuck-at-one'
#error_type = 'stuck-at-zero'
# error rates: 

list_of_frate = [0.0013, 0.0064, 0.0128, 0.1276, 0.1913, 0.2551, 0.3189, 0.3827]

frate = list_of_frate[7]

def insert_faults(state, key, frate):
    # print(state['state_dict']['module.features.0.weight'].data[0, 0, 0])
    print(list(state['state_dict'][key].size()))
    weight_size = list(state['state_dict'][key].size())
    n_indexes = len(weight_size)
    n_elem = torch.numel(state['state_dict'][key])
    n_faults = int(n_elem*frate)
    i = 0
    while (i < n_faults):
        rand_indexes = np.random.randint(0, high=n_elem, size=n_indexes)
        # print(rand_indexes)
        findexes = tuple(np.remainder(rand_indexes, weight_size))
        # if findexes not in (list_findexes):
        i += 1
        # print((findexes))
        # print(state['state_dict'][key].data)
        # input("--------")
        # print(state['state_dict'][key].data[findexes])
        try:
            if error_type == 'stuck-at-zero':
                state['state_dict'][key].data[findexes] = 0.0
            elif error_type == 'stuck-at-one':
                if (state['state_dict'][key].data[findexes] < 0.5 and state['state_dict'][key].data[findexes] >= 0.0):  
                    state['state_dict'][key].data[findexes] += 0.5
                elif (state['state_dict'][key].data[findexes] < -0.5 and state['state_dict'][key].data[findexes] >= -1.0):  
                    state['state_dict'][key].data[findexes] += 0.5
        except:
            print(rand_indexes)
            print(findexes)
        # print(state['state_dict'][key].data[findexes])
        # input("--------")
    print(i)
    print(n_elem)
    return 0

model = VGG_SNN_STDB(vgg_name = 'VGG16', activation = 'Linear', labels=10, timesteps=200, leak=1.0, default_threshold=1.0, alpha=0.3, beta=0.01, dropout=0.3, kernel_size=3, dataset='CIFAR10')
model = nn.DataParallel(model) 
pretrained_snn = './trained_models/snn/snn_vgg16_cifar10.pth'
state = torch.load(pretrained_snn, map_location='cpu')

# insert_faults(state, 'module.classifier.3.weight', frate)

# exit()
                
cur_dict = model.state_dict()     
for key in state['state_dict'].keys():
    
    if key in cur_dict:
        if (state['state_dict'][key].shape == cur_dict[key].shape):
            insert_faults(state, key, frate)

            # cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
            print('\n Loaded {} from {}'.format(key, pretrained_snn))
        else:
            print('\n Size mismatch {}, size of loaded model {}, size of current model {}'.format(key, state['state_dict'][key].shape, model.state_dict()[key].shape))
    else:
        print('\n Loaded weight {} not present in current model'.format(key))
# model.load_state_dict(cur_dict)

# if 'thresholds' in state.keys():
#     try:
#         if state['leak_mem']:
#             state['leak'] = state['leak_mem']
#     except:
#         pass
#     if state['timesteps']!=200 or state['leak']!=1.0:
#         print('\n Timesteps/Leak mismatch between loaded SNN and current simulation timesteps/leak, current timesteps/leak {}/{}, loaded timesteps/leak {}/{}'.format(timesteps, leak, state['timesteps'], state['leak']))
#     thresholds = state['thresholds']
#     model.module.threshold_update(scaling_factor = 0.7, thresholds=thresholds[:])
# else:
#     print('\n Loaded SNN model does not have thresholds')

# print('\n {}'.format(model))

try:
    os.mkdir('./trained_models/snn/')
except OSError:
    pass 
filename = './trained_models/snn/snn_vgg16_cifar10_'+error_type+'_'+str(frate)+'.pth'
torch.save(state,filename)    
    
