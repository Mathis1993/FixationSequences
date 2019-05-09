import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torchvision import transforms

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
#Center Bias
from utils.Gaussian_Map import gaussian_map

def create_datasets(batch_size):
    
    #transforms
    data_transform = transforms.Compose([ToTensor(),Downsampling(10), Targets2D(100,100,100)])#ExpandTargets(100)])
    
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

import numpy as np

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

train_loader, val_loader, test_loader = create_datasets(128)


#module for center bias
class Center_Bias(nn.Module):
    def __init__(self, gpu=False):
        super(Center_Bias, self).__init__()
        self.sigma = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.conv = nn.Conv2d(3, 1, 1, stride=1, padding=0)
        self.gpu = gpu
        if gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.cuda()
                
    def forward(self, x):
        x = torch.sum(x, dim=1)
        print("sum after axis summing: {}".format(torch.sum(x)))
        print("gaussian map sum: {}".format(torch.sum(gaussian_map(x, self.sigma, self.w, self.gpu))))
        x = gaussian_map(x, self.sigma, self.w, self.gpu) * x
        print("sum after gaussian map: {}".format(torch.sum(x)))
        #print("size is: {}".format(x.size()))
        #print("sd is: {}".format(self.sigma))
        return x


class TestNet(nn.Module):

    def __init__(self, gpu=False):
        super(TestNet, self).__init__()
        #3 input image channels (color-images), 64 output channels,  3x3 square convolution kernel
        #padding to keep dimensions of output at 100x100
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 1, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.center_bias = Center_Bias(gpu)
        self.gpu = gpu
        if gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.cuda()
        
    def forward(self, x):
        print("input sum at beginning of forward pass: {}".format(torch.sum(x)))
        x = functional.relu(self.conv1(x))
        print("input sum after first conv and relu: {}".format(torch.sum(x)))
        x = self.conv1_bn(x)
        print("input sum after first batch normalization: {}".format(torch.sum(x)))
        x = functional.relu(self.conv2(x))
        print("input sum after second conv and relu: {}".format(torch.sum(x)))
        x = self.conv2_bn(x)
        #print("output shape: {}".format(x.size()))
        print("input sum after second batch normalization: {}".format(torch.sum(x)))
        x = functional.relu(self.conv3(x))
        print("input sum after third conv and relu: {}".format(torch.sum(x)))
        x = self.conv3_bn(x)
        print("input sum after third conv batch nomralization: {}".format(torch.sum(x)))
        x = self.conv4(x)
        print("input sum after fourth conv: {}".format(torch.sum(x)))
        x = self.center_bias(x)
        print("input sum after center bias: {}".format(torch.sum(x)))
        print(x[0,0,:,:])
        return x

#initilaize the NN
model = TestNet(True)
model_c = Center_Bias(True)

criterion = nn.PoissonNLLLoss(log_input=True, full=True, reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=0.25)

t = iter(train_loader)
for i, example in enumerate(t): #start at index 0
    # get the inputs
    data = example["image"]
    #print("input sum: {}".format(torch.sum(data)))
    target = example["fixations"]
    #target_locs = example["fixation_locs"]
       
    # clear the gradients of all optimized variables
    optimizer.zero_grad()
    
    data = data.to('cuda')
    target = target.to('cuda')

    output = model_c(data)
    
    output = output.view(-1, target.size()[-1], target.size()[-2])
    loss = criterion(output, target)
    print(loss)
    
    loss.backward()
    optimizer.step()
    
    #k = 0
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)
    #        print(param.grad)
    #        if k == 0:
    #            break
    
    if i == 20:
        break
            
            