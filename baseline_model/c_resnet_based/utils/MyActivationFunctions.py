import torch

def mySigmoid(x, upper_bound, gpu):
    #scale output of sigmoid and add offset to avoid zero output
    offset = torch.Tensor([10]).pow(-5)
    #push offset-tensor to gpu
    if gpu:
        if torch.cuda.is_available():
            offset = offset.to('cuda')
    return torch.sigmoid(x) * upper_bound + offset