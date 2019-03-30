#!/usr/bin/env python
# coding: utf-8

# In[24]:


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


# In[25]:


os.system("data_conversion.py 1")
os.system("Clean_and_split_data.py 1")


# In[26]:


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


# In[27]:


#if torch.cuda.is_available():
#    device = torch.device("cuda")


# In[28]:


def mySigmoid(x, upper_bound):
    #scale output of sigmoid and add offset to avoid zero output
    offset = torch.Tensor([10]).pow(-5)
    #push offset-tensor to gpu
    #offset = offset.to('cuda')
    return torch.sigmoid(x) * upper_bound + offset


# In[29]:


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
#push model to gpu
#model.cuda()
print(model)


# In[30]:


from visdom import Visdom

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom(server="http://130.63.188.108", port=12345)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Iterations',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


# In[31]:


if __name__ == "__main__":
    
    global plotter
    plotter = VisdomLinePlotter(env_name='main')


# In[32]:


optimizer = optim.SGD(model.parameters(), lr=0.00001)


# In[33]:


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
            #push data and targets to gpu
            #data = data.to('cuda')
            #target = target.to('cuda')
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
            iteration = i + 1
            #plot is always appending the newest value, so just give the last item if the list
            plotter.plot('loss', 'train', 'Loss per Iteration', iteration, train_losses[-1])
            

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for i, example in enumerate(val_loader, 0): #start at index 0
            # get the inputs
            data = example["image"]
            #print("input sum: {}".format(torch.sum(data)))
            target = example["fixations"]
            #push data and targets to gpu
            #data = data.to('cuda')
            #target = target.to('cuda')
            #print("target sum: {}".format(torch.sum(target)))
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            #print("output sum: {}".format(torch.sum(output)))
            # calculate the loss
            loss = myLoss(output, target)
            # record validation loss
            valid_losses.append(loss.item())
            plotter.plot('loss', 'val', 'Loss per Iteration', iteration, valid_losses[-1])

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
        
        #plot average loss for this epoch
        plotter.plot('loss', 'train', 'Loss per Epoch', epoch, train_loss)
        plotter.plot('loss', 'val', 'Loss per Epoch', epoch, valid_loss)
        
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


# In[10]:


batch_size = 4
n_epochs = 10

train_loader, test_loader, valid_loader = create_datasets(batch_size)

# early stopping patience; how long to wait after last time validation loss improved.
patience = 20

model, train_loss, valid_loss = train_model(model, batch_size, patience, n_epochs)


# In[ ]:


# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')


# In[9]:


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

print(list(model.parameters()))


# In[ ]:




