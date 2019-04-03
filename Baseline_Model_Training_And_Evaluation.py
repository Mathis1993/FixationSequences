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
from utils.Dataset_And_Transforms import FigrimFillersDataset, Downsampling, ToTensor, ExpandTargets
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


# In[24]:


#Run data-conversion from matlab to python and cleaning/splitting of data
from Data_Conversion import convert_data
convert_data()
from Clean_And_Split_Data import clean_and_split
clean_and_split()
#writes "allImages_unfolded_clean.json" (complete dataset), as well as "allImages_unfolded_train.json", 
#"allImages_unfolded_val.json" and "allImages_unfolded_test.json" (split datasets)to disk


# In[25]:


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


#set global boolean gpu value, so that everything will run on the gpu (if True) or cpu (if false)
if __name__ == "__main__":
    
    global batch_size
    batch_size = int(sys.argv[1])
    
    global lr
    lr = float(sys.argv[2])
    
    global n_epochs
    n_epochs = int(sys.argv[3])
    
    global gpu
    gpu = bool(sys.argv[4])
    #gpu = True


# In[27]:


class TestNet(nn.Module):

    def __init__(self, gpu=False):
        super(TestNet, self).__init__()
        #3 input image channels (color-images), 64 output channels, 3x3 square convolution kernel
        #padding to keep dimensions of output at 100x100
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 1, 3, stride=1, padding=1)
        #scale parameter for the sigmoid function
        self.upper_bound = nn.Parameter(torch.Tensor([1]))
        #make it considered by autograd
        self.upper_bound.requires_grad_()
        self.gpu = gpu
        if gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.cuda()
        
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
        x = mySigmoid(self.conv3(x), abs(self.upper_bound), gpu)
        #print("input sum after last conv and sigmoid: {}".format(torch.sum(x)))
        
        return x

#initilaize the NN
model = TestNet(gpu)

print(model)
#print(torch.cuda.current_device())
#print(torch.cuda.device(0))
#print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))


# In[28]:


#initialize visdom-line-plotter-instances that are available from everywhere within the script   
if __name__ == "__main__":

    global plotter_train
    plotter_train = VisdomLinePlotter(env_name='training', server="http://130.63.188.108", port=9876)
    
    global plotter_val
    plotter_eval = VisdomLinePlotter(env_name='evaluation', server="http://130.63.188.108", port=9876)


# In[29]:


#optimizer with learning rate
optimizer = optim.SGD(model.parameters(), lr=lr)

# In[30]:


def train_model(model, batch_size, patience, n_epochs, gpu):
    
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
            #print("output sum: {}".format(torch.sum(output)))
            # calculate the loss
            loss = myLoss(output, target)
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


#load df to store results in
results = pd.read_csv("results.csv")
#create temporary df (its contents will be appended to "results.csv")
results_cur = pd.DataFrame(columns = ["batch_size", "learning_rate", "mean_accuracy_per_image", "mean_test_loss", "mean_validation_loss", "mean_train_loss"])

# In[31]:


#call the training/validation loop
batch_size #= 32
n_epochs #= 10

train_loader, val_loader, test_loader = create_datasets(batch_size)

# early stopping patience; how long to wait after last time validation loss improved.
patience = 5

model, train_loss, valid_loss = train_model(model, batch_size, patience, n_epochs, gpu)


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
            
            #push data and targets to gpu
            if gpu:
                if torch.cuda.is_available():
                    data = data.to('cuda')
                    target = target.to('cuda')
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = myLoss(output, target)
            # record training loss
            test_losses.append(loss.item())
            #accuracy
            acc_this_batch = 0
            for batch_idx in range(output.size()[0]):
                output_subset = output[batch_idx]
                target_subset = target[batch_idx]
                acc_this_image, _ = accuracy(output_subset, target_subset, gpu)
                acc_per_image.append(acc_this_image)
                acc_this_batch += acc_this_image
            #divide by batch size
            acc_this_batch /= output.size()[0]
            acc_per_batch.append(acc_this_batch)

acc_per_image = np.asarray(acc_per_image)
print("Mean test loss is: {}".format(np.average(test_losses)))
print("Mean accuracy for test set ist: {}".format(np.mean(acc_per_image)))


# In[9]:

#print value of sigmoid scaling parameter (it's the first one, so stop after one iteration)
k = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
        if k == 0:
            break

#print(list(model.parameters()))

#store all the information from this hyperparameter configuration
results_cur.loc[0,"batch_size"] = batch_size
results_cur.loc[0, "learning_rate"] = lr
results_cur.loc[0, "mean_accuracy_per_image"] = np.mean(acc_per_image)
results_cur.loc[0, "mean_test_loss"] = np.average(test_losses)
results_cur.loc[0, "mean_validation_loss"] = np.average(valid_loss)
results_cur.loc[0, "mean_train_loss"] = np.average(train_loss)

#and append results_cur to results
results = results.append(results_cur)

#store results
results.to_csv("results.csv", index=False, header=True)