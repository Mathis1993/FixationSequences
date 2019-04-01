#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np


# In[2]:


#find indices of these values in unflattend activation-tensor
def map_idx(tensor_unfl, idx_fl):
    """
    Takes unflattened 2D-tensor and index of the same flattened 2D-tensor and returns the corresponding index
    of the unflattened tensor.
    """
    #row_number of unflattened tensor is index of flattened tensor // amount of columns of unflattened tensor
    #col_number of unflattened tensor is index of flattened tensor % amount of columns of unflattened tensor
    n_cols = tensor_unfl.size()[-1]
    row_idx_unfl = idx_fl // n_cols
    col_idx_unfl = idx_fl % n_cols
    return (torch.tensor([row_idx_unfl, col_idx_unfl]))


# In[3]:


def accuracy(activations, fixations):
    """
    Calculates the accuracy for one image's activations and its corresponding fixation sequence.
    
    Takes: - activations: tensor of activations of size (1,act_x,act_y) (1 channel)
           - fixations: tensor of fixations sequence of size (x,2)
    
    Returns: - accuracy for this image: 1, if eg 4 fixations, and the locations of these 4 fixations are equal to 
               the 4 biggest activation values. 0.75, if only 3 fixation locations hit one of the four biggest 
               activation values and so on
            - hits for this image: how many fixations led to one of the x biggest activation values?
    """

    ##Accuracy for one image

    #drop unnecessary first dimension of activations (there is only one channel)
    activations = activations.reshape(activations.size()[-2], activations.size()[-1])

    #how many fixations are there?
    num_fix = 0
    for i,j in fixations:
                if (i,j) == (-1000,-1000):
                    break
                num_fix += 1
    #flatten
    activations_f = activations.view(-1)

    #find x largest values and their indices in flattened activation-tensor
    lar_val, lar_val_idx = torch.topk(activations_f, num_fix)

    idx_unfl = []
    for idx_fl in lar_val_idx:
        idx_unfl.append(map_idx(activations, idx_fl.item()))

    #see if they match with fixations indices
    hits = 0
    #does each fixation lead to one of the x biggest activation values?
    for fix in range(num_fix):
        for idx in idx_unfl:
            hits += torch.all(torch.eq(idx,fixations[fix])).item()

    #calcualte proportion of hits
    acc = hits / num_fix
    
    return acc, hits


# In[179]:


#one image: activations are (1,100,100), fixations are (100,2)
#batch size of 4: activations are (4,1,100,100), fixations are (4,100,2)

#list for one accuracy value per image
#acc_values = []
#for batch_idx in range(activations.size()[0]):
#    activations_subset = activations[batch_idx]
#    fixations_subset = fixations[batch_idx]
#    acc_this_image, _ = accuracy(activations_subset, fixations_subset)
#    acc_values.append(acc_this_image)

#acc_values = np.asarray(acc_values)

