import torch
from utils.EarlyStopping import EarlyStopping

#status bar
from tqdm import tqdm
import numpy as np

def train_model(baseline_model, rnn_model, training_id, patience, n_epochs, gpu, plotter_train, plotter_eval, train_loader, val_loader, optimizer, criterion, lr):
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs + 1):
        
        ###################
        # train the model #
        ###################
        
        rnn_model.train() # prep model for training
        t = tqdm(iter(train_loader), desc="[Train on Epoch {}/{}]".format(epoch, n_epochs))
        for i, example in enumerate(t): #start at index 0
            # get the inputs
            image = example["image"]
            inputs = example["inputs"]
            targets = example["targets"]
            
            #push data and targets to gpu
            if gpu:
                if torch.cuda.is_available():
                    image = image.to('cuda')
                    inputs = inputs.to('cuda')
                    targets = targets.to('cuda')

            optimizer.zero_grad()
            
            
            #move images through baselinemodel
            baseline_output = baseline_model(image)
            
            #combine baseline-output and time-step input
            inputs_combined = torch.empty(inputs.size(0), inputs.size(1), inputs.size(2) + baseline_output.size(0), device='cuda:0')
            for j in range(inputs.size(1)):
                inputs_combined[0,j] = torch.cat((baseline_output, inputs[0,j]),0)
            
            #feed combined inputs to rnn (hidden is initialized automatically)
            output, hidden = rnn_model(inputs_combined)
            
            #CrossEntropyLoss with RNN: output should be 
            #[1, number_of_classes, seq_length], while your target should be [1, seq_length].
            #https://discuss.pytorch.org/t/pytorch-lstm-target-dimension-in-calculating-cross-entropy-loss/30398
            #so switch axes from (1,6,10002 to 1,10002,6)
            output = output.permute(0,2,1)
            loss = criterion(output, targets)

            loss.backward()

            optimizer.step()
            
            train_losses.append(loss.item())
            
            #for the first epoch, plot loss per iteration to have a quick overview of the early training phase
            iteration = i + 1
            #plot is always appending the newest value, so just give the last item if the list
            if epoch == 1:
                plotter_train.plot('loss', 'train', 'Loss per Iteration', iteration, train_losses[-1], training_id, lr, 'iteration')
            
            
        ######################    
        # validate the model #
        ######################
        rnn_model.eval() # prep model for evaluation
        t = tqdm(iter(val_loader), desc="[Valid on Epoch {}/{}]".format(epoch, n_epochs))
        for i, example in enumerate(t):
            # get the inputs
            image = example["image"]
            inputs = example["inputs"]
            targets = example["targets"]
            
            #push data and targets to gpu
            if gpu:
                if torch.cuda.is_available():
                    image = image.to('cuda')
                    inputs = inputs.to('cuda')
                    targets = targets.to('cuda')
            
            #move images through baselinemodel
            baseline_output = baseline_model(image)
            
            #combine baseline-output and time-step input
            inputs_combined = torch.empty(inputs.size(0), inputs.size(1), inputs.size(2) + baseline_output.size(0), device='cuda:0')
            for j in range(inputs.size(1)):
                inputs_combined[0,j] = torch.cat((baseline_output, inputs[0,j]),0)
            
            #feed combined inputs to rnn (hidden is initialized automatically)
            output, hidden = rnn_model(inputs_combined)
            
            #switch axes from (1,6,10002 to 1,10002,6)
            output = output.permute(0,2,1)
            loss = criterion(output, targets)
            
            # record validation loss
            valid_losses.append(loss.item())
            
            # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = str(n_epochs)
        
        print_msg = ('[{}/{}] '.format(epoch, epoch_len) +
                     'train_loss: {:.15f} '.format(train_loss) +
                     'valid_loss: {:.15f}'.format(valid_loss))
        
        print(print_msg)
        
        #plot average loss for this epoch
        plotter_eval.plot('loss', 'train', 'Loss per Epoch', epoch, train_loss, training_id, lr, 'epoch')
        plotter_eval.plot('loss', 'val', 'Loss per Epoch', epoch, valid_loss, training_id, lr, 'epoch')
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, rnn_model, training_id, lr)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    name = "checkpoint_id_{}_lr_{}.pt".format(training_id, lr)
    rnn_model.load_state_dict(torch.load("results/" + name))

    return  rnn_model, avg_train_losses, avg_valid_losses