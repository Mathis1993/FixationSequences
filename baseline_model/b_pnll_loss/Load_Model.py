import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torchvision import transforms
#Dataset and Transforms
from utils.Dataset_And_Transforms import FigrimFillersDataset, Downsampling, ToTensor, ExpandTargets, Targets2D
#evaluation
from utils.Evaluate_Baseline import map_idx, accuracy
import numpy as np

def create_datasets(batch_size):
    
    #transforms
    data_transform = transforms.Compose([ToTensor(),Downsampling(10), Targets2D(100,100,100)])#ExpandTargets(100)])
    
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
        self.pool1 = nn.AdaptiveMaxPool2d((50,50))
        self.conv4 = nn.Conv2d(1, 1, 1, stride=1, padding=25)
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
        #x = self.conv1_bn(x)
        #print("input sum after first batch normalization: {}".format(torch.sum(x)))
        x = functional.relu(self.conv2(x))
        #print("input sum after second conv and relu: {}".format(torch.sum(x)))
        #x = self.conv2_bn(x)
        #print("input sum after second batch normalization: {}".format(torch.sum(x)))
        #if scaled by a negative value, we would try to take the ln of negative values in the loss  function
        #(ln is not defined for negative values), so make sure that the scaling parameter is positive
        #x = mySigmoid(self.conv3(x), abs(self.upper_bound), gpu)
        #x = functional.relu(self.conv3(x))
        x = self.conv3(x)
        #print("input sum after last conv and sigmoid: {}".format(torch.sum(x)))
        x = self.pool1(x)
        x = self.conv4(x)
        
        
        return x

#initilaize the NN
model = TestNet(True)


train_loader, val_loader, test_loader = create_datasets(128)

name = "checkpoint_batch_size_128_lr_0.2.pt"
#name = "Lowest_Loss.pt"
def load_model(name):
    model.load_state_dict(torch.load(name))
load_model(name)

criterion = nn.PoissonNLLLoss(log_input=True, full=True, reduction="mean")

#get one feature map and one fixation
def get_results():
    t = iter(test_loader)
    for i, example in enumerate(t): #start at index 0
                # get the inputs
                data = example["image"]
                #print("input sum: {}".format(torch.sum(data)))
                target = example["fixations"]
                #target_locs = example["fixation_locs"]

                data = data.to('cuda')
                target = target.to('cuda')

                output = model(data)
                if i == 0:
                    break
    return output, target

output, target = get_results()
output = output.view(-1,100,100)
loss = criterion(output, target)
print(loss)

#print(torch.topk(output.view(-1), int(torch.sum(target).item())))
#print(torch.topk(target.view(-1), int(torch.sum(target).item())))
#print(output)
#print(output[0,0,0,0])
#print(output[0,0,49,49])


gpu = True

#evaluate the model
def eval_model():
    # to track the training loss as the model trains
    test_losses = []
    #to track the accuracy 
    acc_per_image = []
    acc_per_batch = []
    #track absolute hits
    hit_list = []
    #track number of fixations
    n_fixations = []

    model.eval() # prep model for evaluation
    t = iter(test_loader)
    for i, example in enumerate(t): #start at index 0
        # get the inputs
        data = example["image"]
        #print("input sum: {}".format(torch.sum(data)))
        target = example["fixations"]
        target_locs = example["fixation_locs"]

        #push data and targets to gpu
        if gpu:
            if torch.cuda.is_available():
                data = data.to('cuda')
                target = target.to('cuda')

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        #drop channel-dimension (is only 1) so that outputs will be of same size as targets (batch_size,100,100)
        output = output.view(-1, target.size()[-2], target.size()[-2])

        loss = criterion(output, target)
        # calculate the loss
        #loss = myLoss(output, target)
        # record training loss
        test_losses.append(loss.item())
        #accuracy
        acc_this_batch = 0
        for batch_idx in range(output.size()[0]):
            output_subset = output[batch_idx]
            target_subset = target[batch_idx]
            target_locs_subset = target_locs[batch_idx]
            acc_this_image, hits, num_fix = accuracy(output_subset, target_subset, target_locs_subset, gpu)
            acc_per_image.append(acc_this_image)
            hit_list.append(hits)
            n_fixations.append(num_fix)
            acc_this_batch += acc_this_image
        #divide by batch size
        acc_this_batch /= output.size()[0]
        acc_per_batch.append(acc_this_batch)

    acc_per_image = np.asarray(acc_per_image)
    print("Mean test loss is: {}".format(np.average(test_losses)))
    print("Mean accuracy for test set ist: {}".format(np.mean(acc_per_image)))
    print("Hits: {}".format(sum(hit_list)))
    
eval_model()
