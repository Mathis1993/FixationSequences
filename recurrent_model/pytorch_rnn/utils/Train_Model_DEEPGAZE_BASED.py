import torch
from utils.EarlyStopping import EarlyStopping

#status bar
from tqdm import tqdm
import numpy as np

def train_model(baseline_model, rnn_model, training_id, patience, n_epochs, gpu, plotter_train, plotter_eval, train_loader, val_loader, optimizer, criterion_fixations, criterion_state, lr):
    
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
            fixations = example["fixations"]
            states = example["states"]
            
            #push data and targets to gpu
            if gpu:
                if torch.cuda.is_available():
                    image = image.to('cuda')
                    fixations = fixations.to('cuda')
                    states = states.to('cuda')

            optimizer.zero_grad()
            
            
            #move images through baselinemodel
            context_vector = baseline_model(image)
            
            #make sure, the context vector is flattened out into one dimension
            context_vector = context_vector.view(-1)
            
            #no teacher forcing, just feed the context vector to the RNN for every time step
            #batch-dimension, time step dimension (number of fixations), context vector dimension
            context_vectors_steps = torch.empty(fixations.size(0), fixations.size(1), context_vector.size(0), device='cuda:0')
            for j in range(fixations.size(1)):
                context_vectors_steps[0,j] = context_vector
            
            #feed context vector (the same for each time step) to rnn (hidden is initialized automatically)
            out_fix, out_state = rnn_model(context_vectors_steps)

            #CrossEntropyLoss with RNN: output should be 
            #[1, number_of_classes, seq_length], while your target should be [1, seq_length].
            #https://discuss.pytorch.org/t/pytorch-lstm-target-dimension-in-calculating-cross-entropy-loss/30398
            #so switch axes from (1,6,10000 to 1,10000,6) (assuming a context vector with 10000 entries)
            out_state = out_state.permute(0,2,1)
            
            loss_fixations = criterion_fixations(out_fix, fixations)
            loss_state = criterion_state(out_state, states)
            loss_total = loss_fixations + loss_state

            loss_total.backward()

            optimizer.step()
            
            train_losses.append(loss_total.item())
            
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
            fixations = example["fixations"]
            states = example["states"]
            
            #push data and targets to gpu
            if gpu:
                if torch.cuda.is_available():
                    image = image.to('cuda')
                    fixations = fixations.to('cuda')
                    states = states.to('cuda')

            #move images through baselinemodel
            context_vector = baseline_model(image)
            
            #make sure, the context vector is flattened out into one dimension
            context_vector = context_vector.view(-1)
            
            #no teacher forcing, just feed the context vector to the RNN for every time step
            #batch-dimension, time step dimension (number of fixations), context vector dimension
            context_vectors_steps = torch.empty(fixations.size(0), fixations.size(1), context_vector.size(0), device='cuda:0')
            for j in range(fixations.size(1)):
                context_vectors_steps[0,j] = context_vector
            
            #feed context vector (the same for each time step) to rnn (hidden is initialized automatically)
            out_fix, out_state = rnn_model(context_vectors_steps)

            #switch axes from (1,steps,outputs) to (1,outputs,steps)
            out_state = out_state.permute(0,2,1)
            
            loss_fixations = criterion_fixations(out_fix, fixations)
            loss_state = criterion_state(out_state, states)
            loss_total = loss_fixations + loss_state

            # record validation loss
            valid_losses.append(loss_total.item())
            
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