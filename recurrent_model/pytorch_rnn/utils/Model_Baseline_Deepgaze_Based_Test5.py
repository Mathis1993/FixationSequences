import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision
from utils.Gaussian_Map import gaussian_map

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

        if self.gpu:
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
        #print(self.conv.weight.size())


    def forward(self, x):
        return self.conv(x)



class Test5Net(nn.Module):

    def __init__(self, gpu=False):
        super(Test5Net, self).__init__()
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
        self.cb = CenterBiasSchuett(gpu)
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
        x = self.smooth(x)
        x = self.cb(x)
        #flattening
        x = x.view(x.size(0),-1)
        x = functional.softmax(x, dim=1)
        #print("input sum after Smoothing: {}".format(torch.sum(x)))
        #softmax to obtain probability distribution
        #x = nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)
        #print("input sum after Softmax: {}".format(torch.sum(x)))
        #print("input shape after Softmax: {}".format(x.size()))

        return x