######################################
#Gleichung DeepgazeII-Paper am Ende von S. 6, Baselinemodel ist Gleichverteilung (jedes Pixel hat Wslkt. von 1/10000)
#--> Vorhergesagte Dichte an fixierten Koordinaten auswerten minus Vorhersage des Baselinemodells (Nullmodells)
#an diesen Stellen; dann schauen ob der Gain über das Baselinemodel sig. von 0 verschieden ist
######################################


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
#from utils.Baseline_Model import Baseline_Model
import pandas as pd
from tqdm import tqdm
from utils.Gaussian_Map import gaussian_map
from utils.Gaussian_Map2 import gaussian_map2


#####################
#MODEL AND TEST DATA#
#####################

gpu = True
class MyVGG19(torch.nn.Module):
    def __init__(self):
        super(MyVGG19, self).__init__()
        #leave out final two layers (avgsize-pooling and flattening)
        features = list(torchvision.models.vgg19(pretrained = True).features)
        self.features = nn.ModuleList(features).eval() 
        
    def forward(self, x):
        results = []
        #go through modules
        for ii,model in enumerate(self.features):
            #forward propagation
            x = model(x)
            #take 4th module's activations
            if ii in {28,29,31,32,35}:
                results.append(x)
           # if ii == 15:
           #     break
        #upsample (rescale to same size)
        #x = functional.interpolate(x, size=(100,100), mode="bilinear")
        #only take intermediate features, not overall output
        x = functional.interpolate(results[0], size=(100,100), mode="bilinear")
        for i in range(1,len(results)):
            #upsample (rescale to same size)
            intermediate = functional.interpolate(results[i], size=(100,100), mode="bilinear")
            #append to other output along feature channel dimension
            x = torch.cat((x,intermediate), 1)
        #return combined activations
        return x


class CenterBiasSchuett(nn.Module):
    def __init__(self, gpu=False):
        super(CenterBiasSchuett, self).__init__()
        self.sigmax = nn.Parameter(torch.Tensor([1000]), requires_grad=False)
        self.sigmay = nn.Parameter(torch.Tensor([1000]), requires_grad=False)
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
        self.vgg = MyVGG19()
        #reduce the 576 (512 + 64) channels before upsampling (see forward function) to prevent memory problems
        #self.red_ch = nn.Conv2d(512, 256, 1)
        #3 input image channels (color-images), 64 output channels,  3x3 square convolution kernel
        #padding to keep dimensions of output at 100x100
        #self.convfirst = nn.Conv2d(576,1,1)
        #self.bnfirst = nn.BatchNorm2d(1)
        #self.convsecond = nn.Conv2d(1,1,1)
        self.conv_1 = nn.Conv2d(2560,16,1)
        self.conv_2 = nn.Conv2d(16,32,1)
        self.conv_3 = nn.Conv2d(32,2,1)
        self.conv_4 = nn.Conv2d(2,1,1)
        self.cb = CenterBiasSchuett()
        self.smooth = Smoothing()
        self.gpu = gpu
        if gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.cuda()
        
    def forward(self, x):
        #print("input sum at beginning of forward pass: {}".format(torch.sum(x)))
        x = self.vgg(x)
        #x = functional.relu(self.convfirst(x))
        #x = self.bnfirst(x)
        #x = self.convsecond(x)
        #print("input sum after vgg: {}".format(torch.sum(x)))
        x = functional.relu(self.conv_1(x))
        #print("input sum after 1. conv and relu: {}".format(torch.sum(x)))
        x = functional.relu(self.conv_2(x))
        #print("input sum after 2. conv and relu: {}".format(torch.sum(x)))
        x = functional.relu(self.conv_3(x))
        #print("input sum after 3. conv and relu: {}".format(torch.sum(x)))
        x = self.conv_4(x)
        #print("input sum after 4. conv: {}".format(torch.sum(x)))
        #x = self.cb(x)
        #print("input sum after CB: {}".format(torch.sum(x)))
        #x = self.smooth(x)
        #print("input sum after Smoothing: {}".format(torch.sum(x)))
        #softmax to obtain probability distribution
        #x = nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)
        #print("input shape after Softmax: {}".format(x.size()))
        #flattening
        x = x.view(batch_size, -1)
        x = functional.softmax(x)
        #print("input sum after Softmax: {}".format(torch.sum(x)))
        #print("input sum after Softmax abs: {}".format(torch.sum(abs(x))))
    
        return x

#initilaize the NN
model = TestNet(gpu)



#load the pretrained parameters
name = "RestNet_Intermediate_Features.pt"
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
results.to_csv("likelihoods_vgg_intermediate_features.csv", header=True, index=False)