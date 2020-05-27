#########
#IMPORTS#
#########

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
#Dataset (with Transforms)
from utils.Create_Datasets import create_datasets
#Model
from utils.Baseline_Model import Baseline_Model
import math


#####################
#MODEL AND TEST DATA#
#####################

#initilaize the NN
model = Baseline_Model(gpu=True)
#load the pretrained parameters
name = "Baseline_Model.pt"
model.load_state_dict(torch.load(name))

#create test data
batch_size = 1
_, _, test_loader = create_datasets(batch_size)


#############
#LIKELIHOODS#
#############


###########
#Our Model#
###########

#log_input True
#in the end: mal minus 1 und e hoch (da negative log likelihood) 


############
#Null Model#
############

#log_input False
#in the end: mal minus 1 und e hoch (da negative log likelihood)


#################
#Saturated Model#
#################

#Likelihood for a pixel with 0 fixations
likeli_zero_fix = 1
#Likelihood for a pixel with 1 fixation
likeli_one_fix = math.e ** -1
#Likelihood for a pixel with 2 fixations
likelihood_two_fix = 2 * math.e ** -2
#Likelihood for a pixel with [max_fix==3] fixations
likelihood_three_fix = 27/6 * math.e ** -3


#what is the max number of fixations in one pixel?
#max_fix = 0
#t = iter(test_loader)
#for i, example in enumerate(t): #start at index 0
#    target = example["fixations"]
#    target = target.to('cuda')
#
#    #index doesn't matter here
#    these_fix, _ = torch.topk(target.view(-1), 1)
#    if i == 0:
#        print(these_fix.size())
#    these_fix = these_fix.item()
#    if these_fix > max_fix:
#        max_fix = these_fix




t = iter(test_loader)
for i, example in enumerate(t): #start at index 0
    target = example["fixations"]
    target = target.to('cuda')
likeli_poiss = 0
target_sat = target.view(-1)
for elem in target_fl:
    if int(elem.item()) == 0:
        likeli_sat += likeli_zero_fix
    elif int(elem.item()) == 1:
        likeli_sat += likeli_one_fix
    elif int(elem.item()) == 2:
        likeli_sat += likeli_two_fix
    elif int(elem.item()) == 3:
        likeli_sat += likeli_three_fix

#Test: nn.PoissonNLLLoss(log_input=False, full=True) --> *-1 --> e hoch --> Übereinstimmend?
#nll = nn.PoissonNLLLoss(log_input=False, full=True, reduction='none')
#likeli_torch = 0
#for elem in target_fl:
#    likeli_torch += math.e ** (-1 * nll(elem, elem))

#print(likeli_poiss)
#print(likeli_torch.item())

#print(target - target * torch.log(target + 1e-8))

#in the end: mal minus 1 und e hoch (da negative log likelihood) 