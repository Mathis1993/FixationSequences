import torch

def myLoss(activations, fixations):
    """
    - Poisson-distributed target: number of fixations 
    - Input images must be quadratic
    - Fixation data is downsampled to a grid of the size the activations have
    
    Takes:
        - activations: Tensor of the form (batch_size, nC, activation_width, activation_height), eg (5,1,100,100)
        - fixations: A list of tensors. The list has length batch_size and each tensor contains the fixations for one image
        - img_size: The height (or width) of the quadratic input images
    
    Returns:
        - Loss: Tensor of size batch_size (so the loss for each image in the batch)
    """
    
    #result is going to be a scalar for each activation in the batch
    result = torch.empty(activations.size()[0])
    #result = torch.sum(torch.sum(activations, dim=-1), dim=-1)  # might be faster
    
    #for each activation in the batch
    for batch_idx in range(activations.size()[0]):
        loss = torch.sum(activations[batch_idx,:,:,:])
        
        #the corresponding fixation
        for i,j in fixations[batch_idx]:
            
            #looking out for end-of-sequence-symbol
            if (i,j) == (-1000,-1000):
                #if we hit it, store results and go to next batch_idx
                result[batch_idx] = loss
                break
            
            #activations is of shape [batch_size, feature_channels, activation_width, activation_height]
            #when using the dataloader. In the architecture of the CNN, we have 1 feature channel in the 
            #final layer, so we always only use this axis
            loss = loss - torch.log(activations[batch_idx,0,i,j])
    
        #result[batch_idx] = loss
    
    #return loss summed over the whole mini-batch
    return torch.sum(result)


def myLoss2(activations, fixations):
    
    #fixations have dimension (batch_size, x, y), so make sure that for the activations (batch_size, nC=1, x, y),
    #the nC-dimension (that is 1) is dropped
    activations = activations.view(activations.size()[0], activations.size()[-1], activations.size()[-2])
    
    #offset, to be able to handle activations values that are 0
    eps = 1e-8
    
    loss = activations - fixations * torch.log(activations + eps)
    
    return torch.sum(loss)
    
