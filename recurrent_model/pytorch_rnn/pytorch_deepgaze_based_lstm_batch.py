#!/usr/bin/env python

"""
Train RNN as available in Pytorch.

Usage: pytorch_rnn.py <lr> <n_epochs> <gpu>

Arguments:
1. lr (float): Learning rate used in SGD.
2. n_epochs (int): Amount of iterations over the training/validation dataset.
3. gpu (bool): If to run on GPU (if available).
(Batch size does not appear as an argument as it has to be 1 due to different lengths of fixation sequences)

Examples:
pytorch_rnn.py 0.0001 20 True

"""

from docopt import docopt
from pprint import pprint
import sys

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

#Dataset and Transforms
from utils.Dataset_And_Transforms import FigrimFillersDataset, Downsampling, ToTensor, SequenceModeling
from utils.Create_Datasets import create_datasets    
from utils.Loss_Plot import loss_plot
from utils.Train_Model_DEEPGAZE_BASED import train_model
from utils.MyVisdom import VisdomLinePlotter
from utils.Model_Baseline_Deepgaze_Based_Test5 import Test5Net

def run_lstm(lr, n_epochs, gpu):

    #Create datasets
    train_loader, val_loader, test_loader = create_datasets(batch_size=1, data_transform=transforms.Compose([ToTensor(), Downsampling(10), SequenceModeling()]))

    #Baselinmodel; deepgazeII-based, Test 5
    #initilaize a baseline-model instance
    baseline_model = Test5Net(gpu)
    #load the trained parameters
    name = "baseline_batch_size_8_lr_5e-05.pt"
    baseline_model.load_state_dict(torch.load("results/" + name))

    #Recurrent Model
    #nur x-y-Kodierung (10002, Bild und x plus y) Verständnis von Distanz für das Modell
    input_size = 10000 #CNN context vector (eg 100x100, so flattened out 10000)
    hidden_size = 20 #vllt eher 10-50 Dimensionen

    class MyRNN(nn.Module):
        def __init__(self, input_size, hidden_size, gpu):
            super(MyRNN, self).__init__()
            self.rec_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
            self.fc_fix = nn.Linear(in_features=hidden_size, out_features=2) #x- and y-coordinate
            self.fc_state = nn.Linear(in_features=hidden_size, out_features=3) #sos, eos, during sequence
            self.gpu = gpu
            if gpu:
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    self.cuda()

        def forward(self, inputs):
            output, (hidden, cell) = self.rec_layer(inputs)
            out_fix = self.fc_fix(output)
            out_state = self.fc_state(output)
            return out_fix, out_state

    rnn_model = MyRNN(input_size=input_size, hidden_size=hidden_size, gpu=gpu) #lstm, gru
    #fc-layer, einen für x,y-Koordinaten, einen für state
    #Dann zwei Loss-Functions, Werte zB addieren (einmal MSE für xy, einmal cross entropy für state)

    #Optimizer
    #optimizer = optim.SGD(cnn_rnn.parameters(), lr=lr)
    optimizer = optim.Adam(rnn_model.parameters(), lr=lr)

    #Loss-Functions
    criterion_fixations = nn.MSELoss()
    criterion_state = nn.CrossEntropyLoss()

    #load df to store results in
    results = pd.read_csv("results/results.csv")
    #set training_id: Increment last one by 1 or start at 0 if no previous id is present
    try:
        last_training_id = results.loc[len(results)-1, "training_id"]
        training_id = last_training_id + 1
    except:
        training_id = 0
    #create temporary df (its contents will be appended to "results.csv")
    results_cur = pd.DataFrame(columns = ["training_id", "n_epochs", "learning_rate", "mean_train_loss", "mean_validation_loss"])

    #initialize visdom-line-plotter-instances 
    plotter_train = VisdomLinePlotter(env_name='training', server="http://130.63.188.108", port=9876)
    plotter_eval = VisdomLinePlotter(env_name='evaluation', server="http://130.63.188.108", port=9876)

    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 5

    #Start Training
    model, train_loss, valid_loss = train_model(baseline_model, rnn_model, training_id, patience, n_epochs, gpu, plotter_train, plotter_eval, train_loader, val_loader, optimizer, criterion_fixations, criterion_state, lr)

    #visualize the loss as the network trained
    loss_plot(train_loss, valid_loss, lr, training_id)

    #store all the information from this hyperparameter configuration
    results_cur.loc[0,"training_id"] = training_id
    results_cur.loc[0, "n_epochs"] = n_epochs
    results_cur.loc[0, "learning_rate"] = lr
    results_cur.loc[0, "mean_train_loss"] = np.average(train_loss)
    results_cur.loc[0, "mean_validation_loss"] = np.average(valid_loss)
    #and append results_cur to results
    results = results.append(results_cur)
    #store results
    results.to_csv("results/results.csv", index=False, header=True)