import torch
import torch.nn as nn
import torch.nn.functional as functional
from Gaussian import makeGaussian

class Baseline_Model(nn.Module):

    def __init__(self, gpu=False):
        super(Baseline_Model, self).__init__()
        #3 input image channels (color-images), 64 output channels, 3x3 square convolution kernel
        #padding to keep dimensions of output at 100x100
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 1, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(1)
        self.gauss = nn.Parameter(torch.from_numpy(makeGaussian(100)).float(), requires_grad=False)
        self.pool1 = nn.AdaptiveMaxPool2d((50,50))
        self.conv4 = nn.Conv2d(1, 1, 1, stride=1, padding=0)
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
        #print("output shape: {}".format(x.size()))
        #print("input sum after second batch normalization: {}".format(torch.sum(x)))
        #if scaled by a negative value, we would try to take the ln of negative values in the loss  function
        #(ln is not defined for negative values), so make sure that the scaling parameter is positive
        #x = mySigmoid(self.conv3(x), abs(self.upper_bound), gpu)
        #x = functional.relu(self.conv3(x))
        x = functional.relu(self.conv3(x))
        x = self.conv3_bn(x)
        #print("input sum after last conv and sigmoid: {}".format(torch.sum(x)))
        #x = self.pool1(x)
        x = self.conv4(x)
        x = self.gauss * x
        
        return x