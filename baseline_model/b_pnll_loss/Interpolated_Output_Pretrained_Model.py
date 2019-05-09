#!/usr/bin/env python

"""
Train and Evaluate Baseline Model.

Usage: Baseline_Model_Training_And_Evaluation.py <gpu>

Arguments:
1. gpu (bool): If to run on GPU (if available).

Examples:
Interpolated_Output_Pretrained_Model.py True

"""

from docopt import docopt
from pprint import pprint

if __name__ == '__main__':
    arguments = docopt(__doc__, version='FIXME')
    pprint(arguments)

# coding: utf-8

###################################################################################################
#Call with arguments batch_size (int), lr (float), n_epochs (int) and gpu (bool) from command line#
###################################################################################################

# In[11]:


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torchvision import transforms
import torchvision

import pandas as pd
import math
import os
from skimage import io, transform
import numpy as np
import sys
#status bar
from tqdm import tqdm

#import exactly in this way to make sure that matplotlib can generate
#a plot without being connected to a display 
#(otherwise _tkinter.TclError: couldn't connect to display localhost:10.0)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#Dataset and Transforms
from utils.Dataset_And_Transforms import FigrimFillersDataset, Downsampling, ToTensor, ExpandTargets, Targets2D
#Loss-Function
from utils.MyLoss import myLoss
#Early stopping
from utils.EarlyStopping import EarlyStopping
#Adapted Sigmoid Function
from utils.MyActivationFunctions import mySigmoid
#Live-Plotting with Visdom
from utils.MyVisdom import VisdomLinePlotter
#evaluation
from utils.Evaluate_Baseline import map_idx, accuracy
#Gaussian Smoothing: Center Bias
from Gaussian import makeGaussian


# In[24]:


#Run data-conversion from matlab to python and cleaning/splitting of data
#from Data_Conversion import convert_data
#convert_data()
#from Clean_And_Split_Data import clean_and_split
#clean_and_split()
#writes "allImages_unfolded_clean.json" (complete dataset), as well as "allImages_unfolded_train.json", 
#"allImages_unfolded_val.json" and "allImages_unfolded_test.json" (split datasets)to disk


# In[25]:


def create_datasets(batch_size):
    
    #transforms
    #downsampling by factor of 10 as the images were resized from (1000,1000) to (100,100),
    #so the fixations have to be, too
    data_transform = transforms.Compose([ToTensor(),Downsampling(10), Targets2D(100,100, 100)])
    
    #load split data
    figrim_dataset_train = FigrimFillersDataset(json_file='allImages_unfolded_train.json',
                                        root_dir='figrim/fillerData/Fillers',
                                         transform=data_transform)

    figrim_dataset_val = FigrimFillersDataset(json_file='allImages_unfolded_val.json',
                                        root_dir='figrim/fillerData/Fillers',
                                         transform=data_transform)

    figrim_dataset_test = FigrimFillersDataset(json_file='allImages_unfolded_test.json',
                                        root_dir='figrim/fillerData/Fillers',
                                         transform=data_transform)
    
    #create data loaders
    #set number of threads to 8 so that 8 processes will transfer 1 batch to the gpu in parallel
    dataset_loader_train = torch.utils.data.DataLoader(figrim_dataset_train, batch_size=batch_size, 
                                             shuffle=True, num_workers=8)

    dataset_loader_val = torch.utils.data.DataLoader(figrim_dataset_val, batch_size=batch_size, 
                                                 shuffle=True, num_workers=8)
    
    
    #no shuffling, as to be able to identify which images were processed well/not so well
    dataset_loader_test = torch.utils.data.DataLoader(figrim_dataset_test, batch_size=batch_size, 
                                                 shuffle=False, num_workers=8)
    
    return dataset_loader_train, dataset_loader_val, dataset_loader_test


# In[26]:

#batch size of 1 so that we can store one tensor per image
batch_size = 1

#set extracted arguments given on calling the script from the command line
gpu = bool(sys.argv[1])
#gpu = True


# In[27]:

#import resnet18 (pretrained)
resnet18 = torchvision.models.resnet18(pretrained=True)
#retrieve all layers but the last two
modules=list(resnet18.children())[:-2]
#redefine the network to have all layers but the last two
resnet18=nn.Sequential(*modules) #*list: unpack
#freeze parameters
for p in resnet18.parameters():
    p.requires_grad = False

#add layers
#resnet18.avgpool = nn.AdaptiveAvgPool2d((100,100))
#resnet18.conv_new = nn.Conv2d(256, 1, 1)

#push to gpu
if gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        resnet18.cuda()


model = resnet18

batch_size #= 32

print('Create data sets')
train_loader, val_loader, test_loader = create_datasets(batch_size)


##############
# train data #
##############

outputs = []
targets = []

model.eval() # prep model for evaluation
t = tqdm(iter(train_loader))
for i, example in enumerate(t):
    # get the inputs
    data = example["image"]
    #print("input sum: {}".format(torch.sum(data)))
    target = example["fixations"]

    #push data and targets to gpu
    if gpu:
        if torch.cuda.is_available():
            data = data.to('cuda')
            target = target.to('cuda')

    #print("target sum: {}".format(torch.sum(target)))
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    #interpolate to size (100,100)
    output_interpolated = functional.interpolate(output, size=(100,100), mode="bilinear")
    
    #store as normal float-tensors, not cuda (otherwise problems when reading the data back in as a dataset)
    #output_interpolated = output_interpolated.to("cpu")
    #target = target.to("cpu")
    outputs.append(output_interpolated)
    targets.append(target)
    
    
#save list of output and target tensors    
torch.save(outputs, 'resnet-outputs-train.pt')
torch.save(targets, 'resnet-targets-train.pt')

############
# val data #
############

outputs = []
targets = []

model.eval() # prep model for evaluation
t = tqdm(iter(val_loader))
for i, example in enumerate(t):
    # get the inputs
    data = example["image"]
    #print("input sum: {}".format(torch.sum(data)))
    target = example["fixations"]

    #push data and targets to gpu
    if gpu:
        if torch.cuda.is_available():
            data = data.to('cuda')
            target = target.to('cuda')

    #print("target sum: {}".format(torch.sum(target)))
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    #interpolate to size (100,100)
    output_interpolated = functional.interpolate(output, size=(100,100), mode="bilinear")
    
    #store as normal float-tensors, not cuda (otherwise problems when reading the data back in as a dataset)
    #output_interpolated = output_interpolated.to("cpu")
    #target = target.to("cpu")
    outputs.append(output_interpolated)
    targets.append(target)
    
    
#save list of output and target tensors    
torch.save(outputs, 'resnet-outputs-val.pt')
torch.save(targets, 'resnet-targets-val.pt')


#############
# test data #
#############

outputs = []
targets = []

model.eval() # prep model for evaluation
t = tqdm(iter(test_loader))
for i, example in enumerate(t):
    # get the inputs
    data = example["image"]
    #print("input sum: {}".format(torch.sum(data)))
    target = example["fixations"]

    #push data and targets to gpu
    if gpu:
        if torch.cuda.is_available():
            data = data.to('cuda')
            target = target.to('cuda')

    #print("target sum: {}".format(torch.sum(target)))
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    #interpolate to size (100,100)
    output_interpolated = functional.interpolate(output, size=(100,100), mode="bilinear")
    
    #store as normal float-tensors, not cuda (otherwise problems when reading the data back in as a dataset)
    #output_interpolated = output_interpolated.to("cpu")
    #target = target.to("cpu")
    outputs.append(output_interpolated)
    targets.append(target)
    
    
#save list of output and target tensors    
torch.save(outputs, 'resnet-outputs-test.pt')
torch.save(targets, 'resnet-targets-test.pt')   


#loaded = torch.load('resnet-outputs.pt')
#print(loaded.size())