import torch
from utils.EarlyStopping import EarlyStopping

#status bar
from tqdm import tqdm
import numpy as np

def train_model(model, training_id, patience, n_epochs, gpu, plotter_train, plotter_eval, train_loader, val_loader, optimizer, criterion, lr):
    
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
        
        model.train() # prep model for training
        t = tqdm(iter(train_loader), desc="[Train on Epoch {}/{}]".format(epoch, n_epochs))
        for i, example in enumerate(t): #start at index 0
            # get the inputs
            #image = example["image"]
            #inputs = example["inputs"]
            targets = example["targets"]
            inputs_combined = example["inputs_combined"]
            
            #push data and targets to gpu
            if gpu:
                if torch.cuda.is_available():
                    inputs_combined = inputs_combined.to('cuda')
                    targets = targets.to('cuda')

            optimizer.zero_grad()

            #hidden = torch.zeros(1,1,10002) #if not initilaized, pytorch will do it on its own
            
            #as opposed to the "plain_rnn" model, pytorch takes care of all sequence steps, so no for loop needed
            output, hidden = model(inputs_combined)
            
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
        model.eval() # prep model for evaluation
        t = tqdm(iter(val_loader), desc="[Valid on Epoch {}/{}]".format(epoch, n_epochs))
        for i, example in enumerate(t):
            # get the inputs
            #image = example["image"]
            #inputs = example["inputs"]
            targets = example["targets"]
            inputs_combined = example["inputs_combined"]
            
            #push data and targets to gpu
            if gpu:
                if torch.cuda.is_available():
                    inputs_combined = inputs_combined.to('cuda')
                    targets = targets.to('cuda')
            
            output, hidden = model(inputs_combined)
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
        early_stopping(valid_loss, model, training_id, lr)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    name = "checkpoint_id_{}_lr_{}.pt".format(training_id, lr)
    model.load_state_dict(torch.load("results/" + name))

    return  model, avg_train_losses, avg_valid_losses