import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import os
from skimage import io, transform
import numpy as np
from torchvision import transforms
import torch.nn.functional as functional


class FigrimFillersDataset(Dataset):
    """Figrim fillers dataset."""

    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
            json_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #use orient="split" on storing and reading
        self.figrim_frame = pd.read_json(json_file, orient="split")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.figrim_frame)

    def __getitem__(self, idx):
        img_name = self.figrim_frame.iloc[idx, 0]
        img_cat = self.figrim_frame.iloc[idx, 1]
        img_path = os.path.join(self.root_dir + '/' + img_cat + '/', img_name)
        image = io.imread(img_path)
        fixations = np.array(self.figrim_frame.iloc[idx, 7])
        
        sample = {'image': image, 'fixations': fixations}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Downsampling(object):
    """
    Downsample the fixations to the size of the network's activation so that both can be fed into the loss-funciton.
    """
    
    def __init__(self, factor):
        assert isinstance(factor, int)
        self.factor = factor
    
    def __call__(self, sample):
        image, fixations = sample['image'], sample['fixations']
        
        #downsampling (eg activations (100,100) and original image size (1000,1000): by a factor of 10)
        fixations = fixations / self.factor
        #make sure fixations are of type float
        fixations = fixations.float()
        #rounding
        fixations = torch.floor(fixations)
        #conversion to long for usage as indices
        fixations = fixations.long()
        
        return {'image': image, 'fixations': fixations}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, fixations = sample['image'], sample['fixations']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        #convert from np to torch
        image = torch.from_numpy(image)
        fixations = torch.from_numpy(fixations)
        #make sure we have floats, as necessary for running through the nn later
        image = image.float()
        return {'image': image,
                'fixations': fixations}


class ExpandTargets(object):
    """
    - Make targets have the same length defined by the input argument length.
    - Targets are padded with the number -1000.
    """
    def __init__(self, length):
        assert isinstance(length, int)
        self.length = length
    
    def __call__(self, sample):
        image, fixations = sample['image'], sample['fixations']
        
        #make sure length is greater than the number of rows fixations has
        assert self.length > fixations.size()[0]
        #establish which amount of padding is needed to result in a row-length of length
        pad_amount = self.length - fixations.size()[0]
        #what amount to pad left, right, upwards, downwards
        fixations = functional.pad(fixations, (0,0,0,pad_amount), mode='constant', value=-1000)
        
        return {'image': image, 'fixations': fixations}
    
    
class Targets2D(object):
    """
    - Make Targets have the same size as the feature map the model outputs (eg (100,100)) so that both the model's
      output and the targets can be fed to the Poisson-Loss provided by Pytorch
    - Takes dim1 and dim2, the dimensions for the tensor to be outputted (should be the same as the model's feature map's
      dimensions)
    - Returns a tensor of eg (100,100) where each entry is 0, except for the locations that were fixated. There, the entries
      are going to be 1
    """
    def __init__(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2
        
    def __call__(self, sample):
        image, fixations = sample['image'], sample['fixations']
        
        #create output tensor
        output = torch.zeros(self.dim1, self.dim2)
        
        #change entries corresponding to fixation locations to 1
        for i,j in fixations:
            output[i,j] = 1
        
        return {'image': image, 'fixations': output}