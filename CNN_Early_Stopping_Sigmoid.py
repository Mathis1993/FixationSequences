#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import os
from skimage import io, transform
import numpy as np
from torchvision import transforms

#Dataset and Transforms
from Dataset_And_Transforms import FigrimFillersDataset, Downsampling, ToTensor, ExpandTargets
from MyLoss import myLoss 

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import matplotlib.pyplot as plt

#early stopping
from EarlyStopping import EarlyStopping


# In[2]:


def create_datasets(batch_size):
    
    #transforms
    data_transform = transforms.Compose([ToTensor(),Downsampling(10), ExpandTargets(100)])
    
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
    dataset_loader_train = torch.utils.data.DataLoader(figrim_dataset_train, batch_size=batch_size, 
                                             shuffle=True, num_workers=0)

    dataset_loader_val = torch.utils.data.DataLoader(figrim_dataset_val, batch_size=batch_size, 
                                                 shuffle=True, num_workers=0)

    dataset_loader_test = torch.utils.data.DataLoader(figrim_dataset_test, batch_size=batch_size, 
                                                 shuffle=True, num_workers=0)
    
    return dataset_loader_train, dataset_loader_val, dataset_loader_test


# In[3]:


def mySigmoid(x, upper_bound):
    #scale output of sigmoid and add offset to avoid zero output
    return torch.sigmoid(x) * upper_bound + torch.Tensor([10]).pow(-5)


# In[4]:


class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()
        #3 input image channels (color-images), 64 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 1, 3)
        self.pool1 = nn.AdaptiveAvgPool2d((100,100))
        #scale parameter for the sigmoid function
        self.upper_bound = nn.Parameter(torch.Tensor([1]))
        #make it considered by autograd
        self.upper_bound.requires_grad_()
        
    def forward(self, x):
        #print("input sum at beginning of forward pass: {}".format(torch.sum(x)))
        x = functional.relu(self.conv1(x))
        #print("input sum after first conv and relu: {}".format(torch.sum(x)))
        x = self.conv1_bn(x)
        #print("input sum after first batch normalization: {}".format(torch.sum(x)))
        x = functional.relu(self.conv2(x))
        #print("input sum after second conv and relu: {}".format(torch.sum(x)))
        x = self.conv2_bn(x)
        #print("input sum after second batch normalization: {}".format(torch.sum(x)))
        #if scaled by a negative value, we would try to take the ln of negative values in the loss  function
        #(ln is not defined for negative values), so make sure that the scaling parameter is positive
        x = mySigmoid(self.conv3(x), abs(self.upper_bound))
        #print("input sum after last conv and sigmoid: {}".format(torch.sum(x)))
        x = self.pool1(x)
        #print("input sum after pooling: {}".format(torch.sum(x)))
        
        return x

#initilaize the NN
model = TestNet()
print(model)


# In[5]:


optimizer = optim.SGD(model.parameters(), lr=0.00001)


# In[6]:


def train_model(model, batch_size, patience, n_epochs):
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for i, example in enumerate(train_loader, 0): #start at index 0
            # get the inputs
            data = example["image"]
            #print("data size: {}".format(data.size()))
            target = example["fixations"]
            #print("target size: {}".format(target.size()))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            #print("output size: {}".format(output.size()))
            # calculate the loss
            loss = myLoss(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            print("On iteration {} loss is {:.3f}".format(i+1, loss.item()))

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for i, example in enumerate(val_loader, 0): #start at index 0
            # get the inputs
            data = example["image"]
            #print("input sum: {}".format(torch.sum(data)))
            target = example["fixations"]
            #print("target sum: {}".format(torch.sum(target)))
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            #print("output sum: {}".format(torch.sum(output)))
            # calculate the loss
            loss = myLoss(output, target)
            # record validation loss
            valid_losses.append(loss.item())
            

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = ('[{}/{}] '.format(epoch, epoch_len) +
                     'train_loss: {:.5f} '.format(train_loss) +
                     'valid_loss: {:.5f}'.format(valid_loss))
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return  model, avg_train_losses, avg_valid_losses


# In[ ]:


batch_size = 16
n_epochs = 2

train_loader, test_loader, valid_loader = create_datasets(batch_size)

# early stopping patience; how long to wait after last time validation loss improved.
patience = 20

model, train_loss, valid_loss = train_model(model, batch_size, patience, n_epochs)


# In[9]:


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

print(list(model.parameters()))


# In[ ]:




