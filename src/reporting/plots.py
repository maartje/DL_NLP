import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_epoch_losses(train_losses, val_losses, fname):
    """Plot training and validation loss for each epoch.""" 
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'ro-', color='blue', label='train loss over epoch')
    plt.plot(val_losses, 'ro-', color='red', label='validation loss after epoch')
    plt.xlabel('#epochs')
    plt.ylabel('avg. loss')
    plt.legend()
    plt.savefig(fname)
