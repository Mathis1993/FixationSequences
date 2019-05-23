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
      are going to be the number of fixations in this grid cell
    """
    def __init__(self, dim1, dim2, pad_length):
        self.dim1 = dim1
        self.dim2 = dim2
        self.pad_length = pad_length
        
    def __call__(self, sample):
        image, fixations = sample['image'], sample['fixations']
        
        #create output tensor
        output = torch.zeros(self.dim1, self.dim2)
        
        #change entries corresponding to fixation locations
        for i,j in fixations:
            output[i,j] = output[i,j] + 1
        
        #to carry easily accessible information about the fixated locations (grid cells)
        #extract fixation locations as list
        #locations = []
        #fixations_l = fixations.tolist()
        #for i in range(len(fixations)):
        #    locations.append(tuple(fixations_l[i]))
        
        #to carry easily accessible information about the fixated locations (grid cells)
        #data loader needs everything to be of the same dimensions, so expand fixations
        pad_amount = self.pad_length - fixations.size()[0]
        #what amount to pad left, right, upwards, downwards
        fixations = functional.pad(fixations, (0,0,0,pad_amount), mode='constant', value=-1000)
        
        return {'image': image, 'fixations': output, 'fixation_locs': fixations}
    
    
class NormalizeTargets(object):
    """
    - Each 2D Target entry corresponds to the number of fixation this pixcel got. To use them in the context of Cross Entropy 
     Loss, the values need to be normalized to sum to 1
    - Apply only after "Target2D"
    """

    def __call__(self, sample):
        image, fixations, fixation_locs = sample['image'], sample['fixations'], sample['fixation_locs']

        #normalize to sum of 1
        fixations = fixations / torch.sum(fixations)

        return {'image': image, 'fixations': fixations, 'fixation_locs': fixation_locs}

    
class SequenceModeling(object):
    """
    After ToTensor() and Downsampling(), data is prepared to be fed into a RNN:
    - Input for each time step: A matrix of eg 8 x 10002 for the case of 7 fixations and a resolution of 100x100 pixels 
    (plus a start-of-sequence-token). So rows: sos, fixation and the last one is eos. Columns are one-hot encoding
    of the point in the flattened image that was fixated (or if it's sos or eos).
    - Targets: Indices of fixations in the flattened image and then the eos to give over to NLLLoss().
    NOTE: Inputs are               sos-fix1-fix2-fix3
          Targets are indices of   fix1-fix2-fix3-eos  so that from each input t, t+1 is predicted
    - Image: Returned flattened
    NOTE: The RNNs available in Pytorch expect only one sequential input (not image and a sequential input), so the sequential
    input for every time step (each fixation) is bound together with the image data.
    """
    
    def index_2d_to_index_fl(self, tensor_size, index_2d):
        """
        Takes 2D-Tensor-Size and the index (as a tensor) of one if its entries.
        Returns the index of this entry for the flattened version of the 2D-Tensor.
        """
    
        n_cols = tensor_size[-1]
        idx_row, idx_col = tuple(index_2d)
        #extract values from tensors
        idx_row = idx_row.item()
        idx_col = idx_col.item()

        return idx_row * n_cols + idx_col
    
    def __call__(self, sample):
        image, fixations = sample['image'], sample['fixations']
        
        #3,100,100 | 7,2
        
        #TIME-STEP-INPUTS: SOS and Fixations
        n_classes = image.size(-2) * image.size(-1) + 2 #plus sos- and eos-token; so here 10002

        inputs = torch.zeros(fixations.size(0)+1, n_classes) #0 is number of fixations, 1 is fixations

        #start-of-sequence-token
        inputs[0,0] = 1

        #fixations
        idx_new_targets = []
        for i in range(fixations.size(0)):
            idx_new_target = self.index_2d_to_index_fl(image.size(), fixations[i])
            idx_new_targets.append(idx_new_target)
            inputs[i+1, idx_new_target+1] = 1
        
        
        #TARGETS: Fixations and EOS
        #As WE USE nn.NLLLOSS(), only the INDEX of the target at each time step is needed, not a whole one-hot-vector
        #We have the indices of the fixations already, now add index of eos-token
        idx_new_targets.append(n_classes - 1)

        #list2tensor
        new_targets = torch.LongTensor(idx_new_targets)
        
        
        #IMAGE
        image_fl = image.view(-1)
            
        #COMBINE IMAGE AND TIME STEP INPUTS
        inputs_combined = torch.empty(inputs.size(0), inputs.size(1) + image_fl.size(0))
        for i in range(inputs.size(0)):
            inputs_combined[i] = torch.cat((image_fl, inputs[i]),0)
        
        return {"image": image, "image_fl": image_fl, "inputs": inputs, "inputs_combined": inputs_combined, "targets": new_targets}