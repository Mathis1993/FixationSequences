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
    #result = torch.tensor([row_idx_unfl, col_idx_unfl])
    #if gpu:
    #    if torch.cuda.is_available():
    #        result = result.to('cuda')
    return (row_idx_unfl, col_idx_unfl)


# In[3]:


def accuracy(activations, fixations, fixation_locs, gpu):
    """
    Calculates the accuracy for one image's activations and its corresponding fixation sequence.
    
    Takes: - activations: tensor of activations of size (1,act_x,act_y) (1 channel)
           - fixations: tensor of fixations sequence of size (x,2)
    
    Returns: - accuracy for this image: 1, if eg 4 fixations, and each fixation location is equal to one of the 
               the 4 biggest activation values. So eg 0.75, if only 3 fixation locations hit one of the four biggest 
               activation values and so on
            - hits for this image: how many fixations led to one of the x biggest activation values?
            - number of fixations this image had in this trial
    """

    ##Accuracy for one image
    
    #already done in the training/val/eval loop
    #drop unnecessary first dimension of activations (there is only one channel)
    #activations = activations.reshape(activations.size()[-2], activations.size()[-1])

    #how many fixations are there?
    #num_fix = 0
    #for i,j in fixations:
    #            if (i,j) == (-1000,-1000):
    #                break
    #            num_fix += 1
                
    #extract the fixated locations
    #locations = []
    #for i in range(fixations.size()[0]):
    #    for j in range(fixations.size()[1]):
    #        if fixations[i,j] != 0:
    #            locations.append((i,j))
    
    #extract fixated locations from tensor form
    #locations = []
    #for i in range(len(fixation_locs)):
    #    locations.append(tuple(fixation_locs.tolist()[i]))
                
    #How many fixations are there?
    num_fix = int(torch.sum(fixations).item())
    
    #select only the first num_fix entries of fixation_locs (rest is (-1000,-1000))
    fixation_locs = fixation_locs[0:num_fix]
    
    #extract fixated locations from expanded tensor
    locations = []
    fixations_l = fixation_locs.tolist()
    #everything coming after the indx num_fix is only (-1000,-1000)
    for i in range(num_fix):
        locations.append(tuple(fixations_l[i]))
    
    #flatten activations
    activations_f = activations.view(-1)

    #find x largest values and their indices in flattened activation-tensor
    lar_val, lar_val_idx = torch.topk(activations_f, num_fix, largest=True)
    
    idx_unfl = []
    for idx_fl in lar_val_idx:
        idx_unfl.append(map_idx(activations, idx_fl.item()))

    #see if they match with fixations indices
    #hits = 0
    #does each fixation lead to one of the x biggest activation values?
    #for fix in range(num_fix):
    #    for idx in idx_unfl:
    #        current = torch.all(torch.eq(idx,fixations[fix]))
    #        hits += current.item()
    
    #see if they match with fixations indices
    hits = 0
    
    #print("Fixation locations: {}".format(locations))
    #print("Indices of biggest activations: {}".format(idx_unfl))
    
    #does each fixation lead to one of the x biggest activation values?
    biggest_activation_locs = torch.zeros(activations.size())
    
    for idx in idx_unfl:
        biggest_activation_locs[idx] = 1
    
    if gpu:
        if torch.cuda.is_available():
            biggest_activation_locs = biggest_activation_locs.to('cuda')
    
    hits = int(torch.sum(biggest_activation_locs * fixations).item())
    
    #for fix in locations:
    #    for idx in idx_unfl:
            #convert the tensor holding the fixation value to 
            #current = torch.all(torch.eq(idx, fixations[fix].long()))
            #hits += current.item()
    #        if fix == idx:
    #            hits += 1
    
    #calcualte proportion of hits
    acc = hits / num_fix
    
    return acc, hits, num_fix


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

