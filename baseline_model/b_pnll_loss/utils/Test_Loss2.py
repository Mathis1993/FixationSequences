import torch
from MyLoss import myLoss2, myLoss

fixations = torch.zeros(1, 100, 100)
fix_locs = [(23,23), (45,76), (43,97), (42, 63)]
for i in range(fixations.size()[0]):
    for fix_loc in fix_locs:
        fixations[i,fix_locs] = 1

activations = torch.rand(1, 100, 100)

print(myLoss2(activations, fixations))

###############################################

activations = activations.view(1,1,100,100)
fixations2 = torch.tensor(([[(23,23), (45,76), (43,97), (42, 63), (-1000,-1000)]]))
print(myLoss(activations, fixations2))

###############################################

import torch.nn as nn

activations = activations.view(1,100,100)

criterion = nn.PoissonNLLLoss(log_input=False)

print(criterion(activations, fixations))

###############################################
