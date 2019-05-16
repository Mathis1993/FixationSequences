#!/usr/bin/env python

"""
Train RNN as available in Pytorch.

Usage: pytorch_rnn.py <lr> <n_epochs> <gpu>

Arguments:
1. lr (float): Learning rate used in SGD.
2. n_epochs (int): Amount of iterations over the training/validation dataset.
3. gpu (bool): If to run on GPU (if available).
(Batch size does not appear as an argument as it has to be 1 due to different lenghts of fixation sequences)

Examples:
pytorch_rnn 0.0001 20 True

"""

from docopt import docopt
from pprint import pprint

if __name__ == '__main__':
    arguments = docopt(__doc__, version='FIXME')
    pprint(arguments)

    
#set extract arguments given on calling the script from the command line        
lr = float(sys.argv[1])
n_epochs = int(sys.argv[2])
gpu = bool(sys.argv[3])

    

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
from utils.Dataset_And_Transforms import FigrimFillersDataset, Downsampling, ToTensor, SequenceModeling
from utils.Create_Datasets import create_datasets    
    

#Create datasets
train_loader, val_loader, test_loader = create_datasets(batch_size=1, data_transform=transforms.Compose([ToTensor(),Downsampling(10), SequenceModeling(flatten_image=True)]))

#Model
#input: image and fixation-input together
rnn = nn.RNN(input_size = 40002, hidden_size=10002, num_layers=1)


for i, example in enumerate(dataset_loader_train): #start at index 0
    # get the inputs
    image = example["image"]
    inputs = example["inputs"]
    targets = example["fixations"]
    if i == 0:
        break
#input of shape (seq_len, batch, input_size)