from VGG_Intermediate_Layers_Test1 import run_script1
from VGG_Intermediate_Layers_Test2 import run_script2
from VGG_Intermediate_Layers_Test3 import run_script3
from VGG_Intermediate_Layers_Test4 import run_script4
from VGG_Intermediate_Layers_Test5 import run_script5

lrs = [0.000001, 0.00001, 0.00005, 0.0001]
for lr in lrs:
    run_script1(batch_size=8, lr=lr, n_epochs=4, gpu=True)
    run_script2(batch_size=8, lr=lr, n_epochs=4, gpu=True)
    run_script3(batch_size=8, lr=lr, n_epochs=4, gpu=True)
    run_script4(batch_size=8, lr=lr, n_epochs=4, gpu=True)
    run_script5(batch_size=8, lr=lr, n_epochs=4, gpu=True)