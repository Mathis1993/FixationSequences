import math
import torch

#always with mean 0,0?
def gaussian(x,y,sigma,w):
    #first part of the formula is left out, as not dependent on x and y
    return w * (1/((2*math.pi)*(sigma**2))) * torch.exp(-((x**2)+(y**2))/(2*(sigma**2))) 

def gaussian_map(activations, sigma, w, gpu):
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
            gaussian_map[i,j] = gaussian(abs(i-mean_row),abs(j-mean_col),sigma,w)
    return gaussian_map

#square, uneven activations array
activations = torch.randn(1,1,5,5)
sigma = 1
w = 1
gpu = True
print(gaussian_map(activations, sigma, w, gpu))
#print(gaussian(1,1,torch.Tensor([2]),1))