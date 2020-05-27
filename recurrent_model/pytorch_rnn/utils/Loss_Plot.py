#import exactly in this way to make sure that matplotlib can generate
#a plot without being connected to a display 
#(otherwise _tkinter.TclError: couldn't connect to display localhost:10.0)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def loss_plot(train_loss, valid_loss, lr, training_id):
    #turn interactive mode off, because plot cannot be displayed in console
    plt.ioff()
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Lowest Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    #plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.title("Training and Validation Loss per Epoch", fontsize=20)
    plt.tight_layout()
    #plt.show() #no showing, only saving
    name = "loss_plot_id_{}_lr_{}.png".format(training_id, lr)
    fig.savefig("results/" + name, bbox_inches='tight')
