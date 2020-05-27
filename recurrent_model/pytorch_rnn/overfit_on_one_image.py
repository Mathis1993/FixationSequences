import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torchvision import transforms
import torchvision
import torch.nn.utils.rnn as rnn_utils


import pandas as pd
import math
import os
from skimage import io, transform
import numpy as np
from tqdm import tqdm

#Dataset and Transforms
from utils.Dataset_And_Transforms import FigrimFillersDataset, Downsampling, ToTensor, SequenceModeling
from utils.Create_Datasets import create_datasets    
from utils.Loss_Plot import loss_plot
from utils.Train_Model_DEEPGAZE_BASED import train_model
from utils.MyVisdom import VisdomLinePlotter
from utils.Model_Baseline_Deepgaze_Based_Test5 import Test5Net

batch_size = 1
gpu = True

#Create test set
_, _, test_loader = create_datasets(batch_size=batch_size, data_transform=transforms.Compose([ToTensor(), Downsampling(10), SequenceModeling()]))

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
        self.rec_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True) #output size == hidden size?
        self.fc_fix = nn.Linear(in_features=hidden_size, out_features=2) #x- and y-coordinate
        self.fc_state = nn.Linear(in_features=hidden_size, out_features=3) #sos, eos, during sequence
        self.gpu = gpu
        if gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.cuda()
                
    def forward(self, inputs, length):
        #pack batch so that the rnn only sees not-padded inputs
        packed_inputs = rnn_utils.pack_padded_sequence(input=inputs, lengths=length, batch_first=True)
        output, (hidden, cell) = self.rec_layer(packed_inputs)
        #unpack (reverse operation)
        unpacked_output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True, padding_value=-1)
        out_fix = self.fc_fix(unpacked_output)
        out_state = self.fc_state(unpacked_output)
        return out_fix, out_state

rnn_model = MyRNN(input_size=input_size, hidden_size=hidden_size, gpu=gpu) #lstm, gru
#fc-layer, einen für x,y-Koordinaten, einen für state
#Dann zwei Loss-Functions, Werte zB addieren (einmal MSE für xy, einmal cross entropy für state)

lr = 0.05 #0.05 with 6000 iterations was good for just the coordinate prediction

#Optimizer
#optimizer = optim.SGD(cnn_rnn.parameters(), lr=lr)
optimizer = optim.Adam(rnn_model.parameters(), lr=lr)

#Loss-Functions
criterion_fixations = nn.MSELoss()
#Durch ignore_index keine Maskierung (entfernen der gepaddeten -1en) mehr nötig, da diese Einträge einfach ignoriert werden?
criterion_state = nn.CrossEntropyLoss(ignore_index=-1)

n_epochs = 8000

###################
# train the model #
###################

#test_loader, because not randomized
for i, example in enumerate(test_loader): #start at index 0 
    # get the inputs
    image = example["image"]
    fixations = example["fixations"]
    states = example["states"]
    length = example["fixations_length"]

    #push data and targets to gpu
    if gpu:
        if torch.cuda.is_available():
            image = image.to('cuda')
            fixations = fixations.to('cuda')
            states = states.to('cuda')
            length = length.to('cuda')

    #only use the first image
    if i == 0:
        break

#Sort the sequence lengths in descending order, keep track of the old indices, as the fixations' and token-indices'
#batch-dimension needs to be rearranged in that way
length_s, sort_idx = torch.sort(length, 0, descending=True)
#make length_s to list
length_s = list(length_s)
#rearrange batch-dimensions (directly getting rid of the additional dimension this introduces)
fixations = fixations[sort_idx].view(fixations.size(0), fixations.size(-2), fixations.size(-1))
states = states[sort_idx].view(states.size(0), states.size(-1))
#Da der Input immer derselbe Kontextvektor ist, macht es nichts, wenn die Targets umsortiert werden

with torch.no_grad():
    #move images through baseline model
    context_vector = baseline_model(image)

#make sure, the context vector is flattened out into one dimension (and batch-dimension)
context_vector = context_vector.view(context_vector.size(0), -1)

#no teacher forcing, just feed the context vector to the RNN for every time step
#batch-dimension, time step dimension (number of fixations), context vector dimension
                                    #so that all dims are in int
context_vectors_steps = torch.empty(fixations.size(0), int(max(length).item()), context_vector.size(1), device='cuda:0') 
for j in range(fixations.size(0)): #over batch-dimension
    for k in range(int(max(length).item())): #and through rows
        context_vectors_steps[j,k] = context_vector[j] #make every row to be the context vector
        
rnn_model.train() # prep model for training
for epoch in tqdm(range(1, n_epochs + 1)):
       
        #zero out gradients (to prevent them from accumulating)
        optimizer.zero_grad()

        #feed context vector (the same for each time step) to rnn (hidden is initialized automatically)
        out_fix, out_state = rnn_model(context_vectors_steps, length_s)

        #CrossEntropyLoss with RNN: output should be 
        #[1, number_of_classes, seq_length], while your target should be [1, seq_length].
        #https://discuss.pytorch.org/t/pytorch-lstm-target-dimension-in-calculating-cross-entropy-loss/30398
        #so switch axes from (1,6,10000 to 1,10000,6) (assuming a output vector with 10000 entries)
        out_state = out_state.permute(0,2,1)

        #Cut off Sequence Length of fixations and states from 15 to the max sequence length in this batch
        fixations = fixations[:,:out_fix.size(1),:]
        states = states[:,:out_state.size(2)]

        #Mask padded parts in output and targets (fixations): Select (by index) only the activations that correspond
        #to unpadded entries and only feed them to the loss function. Done in flattened out way. Hopefully not necessary
        #for token indices as using argument "ignore_index" with CrossEntropyLoss
        mask_fix = (fixations != -1)
        mask_states = (states != -1)

        masked_out_fix = out_fix[mask_fix] #this also flattens the tensor out
        masked_fixations = fixations[mask_fix]
        masked_states = states[mask_states]

        #Calcuale losses
        loss_fixations = criterion_fixations(masked_out_fix, masked_fixations)
        loss_state = criterion_state(out_state, states)
        loss_total = loss_fixations + loss_state

        #backprop
        loss_total.backward()

        #parameter update
        optimizer.step()
        
        #print loss every 200 iterations
        if epoch % 200 == 0:
            print(loss_total.item())
            

#then make prediction for that image
##################  
# test the model #
##################
rnn_model.eval() # prep model for evaluation

#feed context vector (the same for each time step) to rnn (hidden is initialized automatically)
out_fix, out_state = rnn_model(context_vectors_steps, length_s)

#switch axes from (1,steps,outputs) to (1,outputs,steps)
#out_state = out_state.permute(0,2,1)

#Cut off Sequence Length of fixations and states from 15 to the max sequence length in this batch
fixations = fixations[:,:out_fix.size(1),:]
states = states[:,:out_state.size(2)]

#Mask padded parts in output and targets (fixations): Select (by index) only the activations that correspond
#to unpadded entries and only feed them to the loss function. Done in flattened out way. Hopefully not necessary
#for token indices as using argument "ignore_index" with CrossEntropyLoss
mask_fix = (fixations != -1)
mask_states = (states != -1)

masked_out_fix = out_fix[mask_fix] #this also flattens the tensor out
masked_fixations = fixations[mask_fix]

print(out_state.size())
print(out_state)
print(masked_out_fix.size())
print(masked_out_fix)
print(masked_fixations.size())
print(masked_fixations)