import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision

class MyVGG19(torch.nn.Module):
    def __init__(self, gpu):
        super(MyVGG19, self).__init__()
        #leave out final two layers (avgsize-pooling and flattening)
        features = list(torchvision.models.vgg19(pretrained = True).features)
        self.features = nn.ModuleList(features).eval()
        self.gpu = gpu
        if gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.cuda()

    def forward(self, x):
        results = []
        #go through modules
        for ii,model in enumerate(self.features):
            #forward propagation
            x = model(x)
        return x