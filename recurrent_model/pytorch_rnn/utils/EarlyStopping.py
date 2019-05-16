import numpy as np
import torch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Taken from Github: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, training_id, lr):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, training_id, lr)
        elif score < self.best_score:
            self.counter += 1
            #f-strings only available from python 3.6 onwards
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, training_id, lr)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, training_id, lr):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            #print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))
        name = "checkpoint_id_{}_lr_{}.pt".format(training_id, lr)
        torch.save(model.state_dict(), "results/" + name)
        self.val_loss_min = val_loss