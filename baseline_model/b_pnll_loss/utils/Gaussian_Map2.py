import math
import torch

#always with mean 0,0?
def gaussian(x,y):
    #first part of the formula is left out, as not dependent on x and y
    return ((x**2)+(y**2))/100 

def gaussian_map2(activations, gpu):
    """
    Takes:
    - activations: activations array (last two dimensions are used, these have to be odd and square)
    - sigma: standard deviation for 2D-Gauss-Function (mean is always (0,0)) [Tensor with one scalar]
    - w: weight for scaling the value of the 2D-Gauss-Function
    - gpu: bool: Do computations on gpu or cpu 
    Gives:
    - Array of the size of the last two dimensions with the 2D-Gauss-Function evaluated at each array entry
    """
    
    rows = torch.Tensor([activations.size()[-2]])
    cols = torch.Tensor([activations.size()[-1]])
    gaussian_map = torch.zeros(int(rows.item()), int(cols.item()))
    
    if gpu:
        if torch.cuda.is_available():
            gaussian_map = gaussian_map.to('cuda')
            rows = rows.to('cuda')
            cols = cols.to('cuda')
    
    mean_row = rows // 2
    mean_col = cols // 2

    for i in range(int(rows.item())):
        for j in range(int(cols.item())):
            gaussian_map[i,j] = gaussian(abs(i-mean_row),abs(j-mean_col))
    
    #gaussian_map = (gaussian_map - torch.mean(activations)) / torch.std(activations)
    #print("Sum of gaussian_map: {}".format(torch.sum(gaussian_map)))
    
    #Now, we have a (100,100) gaussian map. Has to inflated to match the activation's dimensions (batch size and one feature 
    #channel).
    #Add feature channel and batch-dimension
    gaussian_map = gaussian_map.view(1,1,gaussian_map.size()[-2], gaussian_map.size()[-1])
    #repeat in the batch size dimension so often that it matches the activations' batch size
    #repeat eg 128 times in first dimension, 1 time (no change) in the others
    #gaussian_map = gaussian_map.repeat(activations.size()[0],1,1,1) 
    
    return gaussian_map

#square, uneven activations array
#activations = torch.randn(128,1,5,5)
#a = 1
#gpu = False
#print(gaussian_map2(activations, a, gpu))
#print(gaussian(1,1,torch.Tensor([2]),1))
#out = torch.cat((activations,gaussian_map2(activations, a, gpu)),1)
#print(out[0,1])
#print(out[1,1])
#print(out.size())