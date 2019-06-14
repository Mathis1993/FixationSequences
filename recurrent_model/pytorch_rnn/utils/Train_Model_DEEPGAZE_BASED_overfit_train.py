import torch
import torch.nn.utils.rnn as rnn_utils
from utils.EarlyStopping import EarlyStopping

#status bar
from tqdm import tqdm
import numpy as np

def train_model(baseline_model, rnn_model, training_id, patience, n_epochs, gpu, plotter_train, plotter_eval, train_loader, val_loader, optimizer, criterion_fixations, lr):
    
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
            length = example["fixations_length"]
            fix_tf = example["fixations_tf"]
            
            #push data and targets to gpu
            if gpu:
                if torch.cuda.is_available():
                    image = image.to('cuda')
                    fixations = fixations.to('cuda')
                    length = length.to('cuda')
                    fix_tf = fix_tf.to('cuda')
                    
            #zero out gradients (to prevent them from accumulating)
            optimizer.zero_grad()
            
            #Sort the sequence lengths in descending order, keep track of the old indices, as the fixations' and token-indices'
            #batch-dimension needs to be rearranged in that way
            length_s, sort_idx = torch.sort(length, 0, descending=True)
            #make length_s to list
            length_s = list(length_s)
            #rearrange batch-dimensions (directly getting rid of the additional dimension this introduces)
            fixations = fixations[sort_idx].view(fixations.size(0), fixations.size(-2), fixations.size(-1))
            #Da der Input immer derselbe Kontextvektor ist, macht es nichts, wenn die Targets umsortiert werden
            
            #move images through baseline model
            with torch.no_grad():
                context_vector = baseline_model(image)
            
            #make sure the context vector is flattened out into one dimension (and batch-dimension)
            context_vector = context_vector.view(context_vector.size(0), -1)
            
            ##combine image and fixation data for teacher forcing
            #batch-dimension, time step dimension (number of fixations), context vector and previous fixation dimension
                                                #so that all dims are in int
            context_vectors_steps = torch.empty(fixations.size(0), int(max(length).item()), context_vector.size(1) + 2, device='cuda:0') #plus 2 for previous x- and y-coordinate
            for j in range(fixations.size(0)): #over batch-dimension
                for k in range(int(max(length).item())): #and through rows
                    #last time step is not given as input as fix_tf has one more row than the max(length)
                    context_vectors_steps[j,k] = torch.cat((context_vector[j], fix_tf[j,k]), 0) #make every row to be the context vector
            
            #feed context vector (the same for each time step) to rnn (hidden is initialized automatically)
            out_fix = rnn_model(context_vectors_steps, length_s)
            
            #Cut off Sequence Length of fixations and states from 15 to the max sequence length in this batch
            fixations = fixations[:,:int(max(length).item()),:]
            
            #Mask padded parts in output and targets (fixations): Select (by index) only the activations that correspond
            #to unpadded entries and only feed them to the loss function. Done in flattened out way.
            mask_fix = (fixations != -1)
            
            masked_out_fix = out_fix[mask_fix] #this also flattens the tensor out
            masked_fixations = fixations[mask_fix]
            
            #Calcuale losses
            loss_fixations = criterion_fixations(masked_out_fix, masked_fixations)
            
            #backprop
            loss_fixations.backward()
            
            #parameter update
            optimizer.step()
            
            train_losses.append(loss_fixations.item())
            
            #for the first epoch, plot loss per iteration to have a quick overview of the early training phase
            iteration = i + 1
            #plot is always appending the newest value, so just give the last item if the list
            if epoch == 1:
                plotter_train.plot('loss', 'train', 'Loss per Iteration', iteration, train_losses[-1], training_id, lr, 'iteration')
            
            
        ######################    
        # validate the model # #HERE ALSO TRAIN_LOADER, AS WE WANT TO OVERFIT ON THE TRAINING DATA
        ######################
        rnn_model.eval() # prep model for evaluation
        t = tqdm(iter(train_loader), desc="[Valid on Epoch {}/{}]".format(epoch, n_epochs))
        for i, example in enumerate(t):
            # get the inputs
            image = example["image"]
            fixations = example["fixations"]
            length = example["fixations_length"]
            fix_tf = example["fixations_tf"]
            
            #push data and targets to gpu
            if gpu:
                if torch.cuda.is_available():
                    image = image.to('cuda')
                    fixations = fixations.to('cuda')
                    length = length.to('cuda')
                    fix_tf = fix_tf.to('cuda')
            
            #Sort the sequence lengths in descending order, keep track of the old indices, as the fixations' and token-indices'
            #batch-dimension needs to be rearranged in that way
            length_s, sort_idx = torch.sort(length, 0, descending=True)
            #make length_s to list
            length_s = list(length_s)
            #rearrange batch-dimensions (directly getting rid of the additional dimension this introduces)
            fixations = fixations[sort_idx].view(fixations.size(0), fixations.size(-2), fixations.size(-1))
            #Da der Input immer derselbe Kontextvektor ist, macht es nichts, wenn die Targets umsortiert werden
            
            #move images through baselinemodel
            with torch.no_grad():
                context_vector = baseline_model(image)
            
            #make sure the context vector is flattened out into one dimension (and batch-dimension)
            context_vector = context_vector.view(context_vector.size(0), -1)
            
            ##combine image and fixation data for teacher forcing
            #batch-dimension, time step dimension (number of fixations), context vector and previous fixation dimension
                                                #so that all dims are in int
            context_vectors_steps = torch.empty(fixations.size(0), int(max(length).item()), context_vector.size(1) + 2, device='cuda:0') #plus 2 for previous x- and y-coordinate
            for j in range(fixations.size(0)): #over batch-dimension
                for k in range(int(max(length).item())): #and through rows
                    #last time step is not given as input as fix_tf has one more row than the max(length)
                    context_vectors_steps[j,k] = torch.cat((context_vector[j], fix_tf[j,k]), 0) #make every row to be the context vector
            
            #feed context vector (the same for each time step) to rnn (hidden is initialized automatically)
            out_fix = rnn_model(context_vectors_steps, length_s)
            
            #Cut off Sequence Length of fixations and states from 15 to the max sequence length in this batch
            fixations = fixations[:,:int(max(length).item()),:]
            
            #Mask padded parts in output and targets (fixations): Select (by index) only the activations that correspond
            #to unpadded entries and only feed them to the loss function. Done in flattened out way.
            mask_fix = (fixations != -1)
            
            masked_out_fix = out_fix[mask_fix] #this also flattens the tensor out
            masked_fixations = fixations[mask_fix]
            
            #Calcuale losses
            loss_fixations = criterion_fixations(masked_out_fix, masked_fixations)

            # record validation loss
            valid_losses.append(loss_fixations.item())
            
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