#!/usr/bin/env python

"""
Train and Evaluate Baseline Model.

Usage: Baseline_Model_Training_And_Evaluation.py <batch_size> <lr> <n_epochs> <gpu>

Arguments:
1. batch_size (int): Size of the batches used for training, validating and testing the network.
2. lr (float): Learning rate used in SGD.
3. n_epochs (int): Amount of iterations over the training/validation dataset.
4. gpu (bool): If to run on GPU (if available).

Examples:
Baseline_Model_Training_And_Evaluation.py 32 0.0001 20 True

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
from utils.Gaussian_Map2 import gaussian_map2
from utils.Gaussian_Map import gaussian_map

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


#set extract arguments given on calling the script from the command line
    
batch_size = int(sys.argv[1])
    
lr = float(sys.argv[2])
    
n_epochs = int(sys.argv[3])
    
gpu = bool(sys.argv[4])
#gpu = True


# In[27]:

#import resnet18 (pretrained)
resnet18 = torchvision.models.resnet18(pretrained=True)
#retrieve all layers but the last two
modules=list(resnet18.children())[:-2]
#redefine the network to have all layers but the last two
resnet18=nn.Sequential(*modules) #*list: unpack
#freeze parameters
#for p in resnet18.parameters():
#    p.requires_grad = False

#add layers
#resnet18.avgpool = nn.AdaptiveAvgPool2d((100,100))
#resnet18.conv_new = nn.Conv2d(256, 1, 1)

#push to gpu
if gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        resnet18.cuda()

#fast identisch zu utils/Gaussian_Map.py
class CenterBiasSchuett(nn.Module):
    def __init__(self, gpu=False):
        super(CenterBiasSchuett, self).__init__()
        self.sigmax = nn.Parameter(torch.Tensor([1000]), requires_grad=True)
        self.sigmay = nn.Parameter(torch.Tensor([1000]), requires_grad=True)
        self.gpu = gpu
    
    def forward(self, x):
        #eg input dimension of 100: tensor with entries 0-99
        gridx = torch.Tensor(range(x.size()[-2]))
        gridy = torch.Tensor(range(x.size()[-1]))
        
        if gpu:
            if torch.cuda.is_available():
                gridx = gridy.to('cuda')
                gridy = gridy.to('cuda')
        
        gridx = gridx - torch.mean(gridx)
        gridx = gridx ** 2
        gridx = gridx / (self.sigmax ** 2)
        gridx = gridx.view(gridx.size()[0],1)
        
        gridy = gridy - torch.mean(gridy)
        gridy = gridy ** 2
        gridy = gridy / (self.sigmay ** 2)
        
        grid = gridx + gridy
        CB = torch.exp(-0.5*grid)
        CB = CB / torch.sum(CB)
        return x * CB
    

class Smoothing(nn.Module):
    def __init__(self, gpu=False):
        super(Smoothing, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3, bias=False, padding=1)
        self.gpu = gpu
        self.conv.weight = torch.nn.Parameter(gaussian_map(torch.rand(3,3), 1, 1, self.gpu).view(1,1,3,3), requires_grad=False)
        print(self.conv.weight.size())
    
    def forward(self, x):
        return self.conv(x)
        

class TestNet(nn.Module):

    def __init__(self, gpu=False):
        super(TestNet, self).__init__()
        self.resnet = resnet18
        #reduce the 512 channels before upsampling (see forward function) to prevent memory problems
        self.red_ch = nn.Conv2d(512, 256, 1)
        #3 input image channels (color-images), 64 output channels,  3x3 square convolution kernel
        #padding to keep dimensions of output at 100x100
        self.convfirst = nn.Conv2d(512,1,1)
        self.bnfirst = nn.BatchNorm2d(1)
        self.convsecond = nn.Conv2d(1,1,1)
        self.conv1 = nn.Conv2d(512, 128, 3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(1)
        #self.pool1 = nn.AdaptiveMaxPool2d((50,50))
        self.conv4 = nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.conv4_bn = nn.BatchNorm2d(1)
        self.center_bias_schuett = CenterBiasSchuett(gpu)
        self.smoothing = Smoothing(gpu)
        self.gpu = gpu
        if gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.cuda()
        
    def forward(self, x):
        x = self.resnet(x)
        #print(x.size())
        #x = self.red_ch(x)
        #upsample resnet-output to 100x100
        x = functional.interpolate(x, size=(100,100), mode="bilinear")
        x = functional.relu(self.convfirst(x))
        x = self.bnfirst(x)
        x = self.convsecond(x)
        #print("input sum at beginning of forward pass: {}".format(torch.sum(x)))
        #x = functional.relu(self.conv1(x))
        #print("input sum after first conv and relu: {}".format(torch.sum(x)))
        #x = self.conv1_bn(x)
        #print("input sum after first batch normalization: {}".format(torch.sum(x)))
        #x = functional.relu(self.conv2(x))
        #print("input sum after second conv and relu: {}".format(torch.sum(x)))
        #x = self.conv2_bn(x)
        #print("output shape: {}".format(x.size()))
        #print("input sum after second batch normalization: {}".format(torch.sum(x)))
        #x = functional.relu(self.conv3(x))
        #x = self.conv3_bn(x)
        #print("input sum after last conv and sigmoid: {}".format(torch.sum(x)))
        #x = self.conv4(x)
        #x = self.conv4_bn(x)
        #x = self.center_bias_schuett(x)
        #x = self.smoothing(x)
        
        return x

#initilaize the NN
model = TestNet(gpu)

print(model)
#print(torch.cuda.current_device())
#print(torch.cuda.device(0))
#print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))


# In[28]:


#initialize visdom-line-plotter-instances 

plotter_train = VisdomLinePlotter(env_name='training', server="http://130.63.188.108", port=9876)
    
plotter_eval = VisdomLinePlotter(env_name='evaluation', server="http://130.63.188.108", port=9876)


# In[29]:


#optimizer with learning rate
optimizer = optim.Adam(model.parameters(), lr=lr)

#lr-scheduler
#lambda1 = lambda epoch: 0.9 ** epoch
#scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
#if lr is 0.2 in the beginning, after 60 epochs it will decrease to 0.02 with gamma = 0.1
#scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

#Poisson-Loss
criterion = nn.PoissonNLLLoss(log_input=True, full=True, reduction="mean")

# In[30]:


def train_model(model, batch_size, patience, n_epochs, gpu, plotter_train, plotter_eval):
    
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
        
        #schedule LR
        #scheduler.step()
        
        model.train() # prep model for training
        t = tqdm(iter(train_loader), desc="[Train on Epoch {}/{}]".format(epoch, n_epochs))
        for i, example in enumerate(t): #start at index 0
            # get the inputs
            data = example["image"]
            #print("data size: {}".format(data.size()))
            target = example["fixations"]
            
            #print("target size: {}".format(target.size()))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            
            #push data and targets to gpu
            if gpu:
                if torch.cuda.is_available():
                    data = data.to('cuda')
                    target = target.to('cuda')
            
            #print(resnet18(data).size())
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            #print(output.size())
            
            #drop channel-dimension (is only 1) so that outputs will be of same size as targets (batch_size,100,100)
            #infer batch dimension as last batch won't have the full size of eg 128
            output = output.view(-1, target.size()[-1], target.size()[-2])
            
            #print("output size: {}".format(output.size()))
            # calculate the loss
            #loss = myLoss(output, target)
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            #print("On iteration {} loss is {:.3f}".format(i+1, loss.item()))
            #for the first epoch, plot loss per iteration to have a quick overview of the early training phase
            iteration = i + 1
            #plot is always appending the newest value, so just give the last item if the list
            if epoch == 1:
                plotter_train.plot('loss', 'train', 'Loss per Iteration', iteration, train_losses[-1], batch_size, lr, 'iteration')
            
            

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        t = tqdm(iter(val_loader), desc="[Valid on Epoch {}/{}]".format(epoch, n_epochs))
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
            
            #drop channel-dimension (is only 1) so that outputs will be of same size as targets (batch_size,100,100)
            output = output.view(-1, target.size()[-2], target.size()[-2])
            
            #print("output sum: {}".format(torch.sum(output)))
            # calculate the loss
            #loss = myLoss(output, target)
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())
            #plotter_val.plot('loss', 'val', 'Loss per Iteration', iteration, valid_losses[-1])

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = str(n_epochs)
        
        print_msg = ('[{}/{}] '.format(epoch, epoch_len) +
                     'train_loss: {:.5f} '.format(train_loss) +
                     'valid_loss: {:.5f}'.format(valid_loss))
        
        print(print_msg)
        
        #plot average loss for this epoch
        plotter_eval.plot('loss', 'train', 'Loss per Epoch', epoch, train_loss, batch_size, lr, 'epoch')
        plotter_eval.plot('loss', 'val', 'Loss per Epoch', epoch, valid_loss, batch_size, lr, 'epoch')
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model, batch_size, lr)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    name = "checkpoint_batch_size_{}_lr_{}.pt".format(batch_size, lr)
    model.load_state_dict(torch.load(name))

    return  model, avg_train_losses, avg_valid_losses

print('Starting to load results data')

#load df to store results in
results = pd.read_csv("results.csv")
#create temporary df (its contents will be appended to "results.csv")
results_cur = pd.DataFrame(columns = ["batch_size", "n_epochs", "learning_rate", "mean_accuracy_per_image", "mean_test_loss", "mean_validation_loss", "mean_train_loss", "number_of_hits", "number_of_test_images", "number_of_fixations"])

# In[31]:


#call the training/validation loop
batch_size #= 32
n_epochs #= 10

print('Create data sets')
train_loader, val_loader, test_loader = create_datasets(batch_size)

# early stopping patience; how long to wait after last time validation loss improved.
patience = 50

print('Start trainging')
model, train_loss, valid_loss = train_model(model, batch_size, patience, n_epochs, gpu, plotter_train, plotter_eval)


# In[ ]:


# visualize the loss as the network trained

#turn interactive mode off, because plot cannot be displayed in console
plt.ioff()

fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Lowest Validation Loss')

plt.xlabel('epochs')
plt.ylabel('loss')
#plt.ylim(0, 0.5) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.title("Training and Validation Loss per Epoch", fontsize=20)
plt.tight_layout()
#plt.show() #no showing, only saving
name = "loss_plot_batch_size_{}_lr_{}.png".format(batch_size, lr)
fig.savefig(name, bbox_inches='tight')


# In[ ]:


#evaluate the model
# to track the training loss as the model trains
test_losses = []
#to track the accuracy 
acc_per_image = []
acc_per_batch = []
#track absolute hits
hit_list = []
#track number of fixations
n_fixations = []

######################    
# evaluate the model #
######################
model.eval() # prep model for evaluation
t = tqdm(iter(test_loader), desc="Evaluating Model")
for i, example in enumerate(t): #start at index 0
            # get the inputs
            data = example["image"]
            #print("input sum: {}".format(torch.sum(data)))
            target = example["fixations"]
            target_locs = example["fixation_locs"]
            
            #push data and targets to gpu
            if gpu:
                if torch.cuda.is_available():
                    data = data.to('cuda')
                    target = target.to('cuda')
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            
            #drop channel-dimension (is only 1) so that outputs will be of same size as targets (batch_size,100,100)
            output = output.view(-1, target.size()[-2], target.size()[-2])
            
            loss = criterion(output, target)
            # calculate the loss
            #loss = myLoss(output, target)
            # record training loss
            test_losses.append(loss.item())
            #accuracy
            acc_this_batch = 0
            for batch_idx in range(output.size()[0]):
                output_subset = output[batch_idx]
                target_subset = target[batch_idx]
                target_locs_subset = target_locs[batch_idx]
                acc_this_image, hits, num_fix = accuracy(output_subset, target_subset, target_locs_subset, gpu)
                acc_per_image.append(acc_this_image)
                hit_list.append(hits)
                n_fixations.append(num_fix)
                acc_this_batch += acc_this_image
            #divide by batch size
            acc_this_batch /= output.size()[0]
            acc_per_batch.append(acc_this_batch)
                
acc_per_image = np.asarray(acc_per_image)
print("Mean test loss is: {}".format(np.average(test_losses)))
print("Mean accuracy for test set ist: {}".format(np.mean(acc_per_image)))


# In[9]:

#print value of sigmoid scaling parameter (it's the first one, so stop after one iteration)
#k = 0
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)
#        if k == 0:
#            break

#print(list(model.parameters()))

#store all the information from this hyperparameter configuration
results_cur.loc[0,"batch_size"] = batch_size
results_cur.loc[0, "n_epochs"] = n_epochs
results_cur.loc[0, "learning_rate"] = lr
results_cur.loc[0, "mean_accuracy_per_image"] = np.mean(acc_per_image)
results_cur.loc[0, "mean_test_loss"] = np.average(test_losses)
results_cur.loc[0, "mean_validation_loss"] = np.average(valid_loss)
results_cur.loc[0, "mean_train_loss"] = np.average(train_loss)
results_cur.loc[0, "number_of_hits"] = sum(hit_list)
results_cur.loc[0, "number_of_test_images"] = len(acc_per_image)
results_cur.loc[0, "number_of_fixations"] = sum(n_fixations)
 
#and append results_cur to results
results = results.append(results_cur)

#store results
results.to_csv("results.csv", index=False, header=True)
