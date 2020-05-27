#########
#IMPORTS#
#########

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
#Dataset (with Transforms)
from utils.Create_Datasets import create_datasets
#Model
from utils.Baseline_Model import Baseline_Model
import pandas as pd
from tqdm import tqdm
from utils.Gaussian_Map2 import gaussian_map2


#####################
#MODEL AND TEST DATA#
#####################

gpu = True
#module for center bias
class Center_Bias(nn.Module):
    def __init__(self, gpu=False):
        super(Center_Bias, self).__init__()
        self.sigma = nn.Parameter(torch.Tensor([100]), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.gpu = gpu

    def forward(self, x):
        x = gaussian_map(x, self.sigma, self.w, gpu) * x
        return x

class Center_Bias2(nn.Module):
    def __init__(self, gpu=False):
        super(Center_Bias2, self).__init__()
        self.a = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.gpu = gpu

    def forward(self, x):
        #concatenate along feature channel dimension
        x = torch.cat((x,gaussian_map2(x, self.a, self.gpu)),1)
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
        self.center_bias2 = Center_Bias2(gpu)
        self.conv5 = nn.Conv2d(2, 1, 1)
        self.gpu = gpu
        if gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.cuda()
        
    def forward(self, x):
        #x = torch.rand(128,3,100,100)
        #x = torch.ones(128,3,100,100)
        #x = x.to('cuda')
        #print("input sum at beginning of forward pass: {}".format(torch.sum(x)))
        x = functional.relu(self.conv1(x))
        #print("input sum after first conv and relu: {}".format(torch.sum(x)))
        x = self.conv1_bn(x)
        #print("input sum after first batch normalization: {}".format(torch.sum(x)))
        x = functional.relu(self.conv2(x))
        #print("input sum after second conv and relu: {}".format(torch.sum(x)))
        x = self.conv2_bn(x)
        #print("output shape: {}".format(x.size()))
        #print("input sum after second batch normalization: {}".format(torch.sum(x)))
        x = functional.relu(self.conv3(x))
        #print("input sum after third conv and relu: {}".format(torch.sum(x)))
        x = self.conv3_bn(x)
        #print("input sum after third batch normalization: {}".format(torch.sum(x)))
        x = self.conv4(x)
        #print("input sum after fourth conv: {}".format(torch.sum(x)))
        x = self.center_bias2(x)
        #print("input sum after center bias: {}".format(torch.sum(x)))
        x = self.conv5(x)
        #print("input sum after last conv: {}".format(torch.sum(x)))
        return x

#initilaize the NN
model = TestNet(gpu=True)


#load the pretrained parameters
name = "checkpoint_batch_size_128_lr_0.1.pt"
model.load_state_dict(torch.load(name))

#create test data
batch_size = 1
_, _, test_loader = create_datasets(batch_size)

#create loss-functions for likelihood-computation
nll_model = nn.PoissonNLLLoss(log_input=True, full=True, reduction='none')
nll_null = nn.PoissonNLLLoss(log_input=False, full=True, reduction='none')
nll_saturated = nn.PoissonNLLLoss(log_input=False, full=True, reduction='none')

#lists to store results
lhs_model = []
lhs_null = []
lhs_saturated = []

#go through test data: calculate likelihood for each of the three models per image and append to respective lists
t = tqdm(iter(test_loader), desc="[Computing Likelihoods")
for i, example in enumerate(t): #start at index 0
    data = example["image"]
    target = example["fixations"]
    data = data.to('cuda')
    target = target.to('cuda')
    
    output = model(data)

    #############
    #LIKELIHOODS#
    #############

    #calculate likelihood for each model per image


    ###########
    #Our Model#
    ###########

    likelihood_model = 0
    target_fl = target.view(-1)
    output_fl = output.view(-1)
    for i in range(output_fl.size()[0]):
        likelihood_model += torch.exp(-1 * nll_model(output_fl[i], target_fl[i])).item()
    lhs_model.append(likelihood_model)


    ############
    #Null Model#
    ############

    likelihood_null = 0
    target_fl = target.view(-1)
    null_prediction = torch.sum(target) / 10000
    for elem in target_fl:
        likelihood_null += torch.exp(-1 * nll_null(null_prediction, elem)).item()
    lhs_null.append(likelihood_null)

    #################
    #Saturated Model#
    #################

    likelihood_saturated = 0
    target_fl = target.view(-1)
    #per item, as we calcualte nll with the loss-function and need to go to positive likelihood before summing
    for elem in target_fl:
        likelihood_saturated += torch.exp(-1 * nll_saturated(elem, elem)).item()
    lhs_saturated.append(likelihood_saturated)
    
results = pd.DataFrame({"lhs_null": lhs_null, "lhs_model": lhs_model, "lhs_saturated": lhs_saturated})
results.to_csv("likelihoods_center_bias.csv", header=True, index=False)