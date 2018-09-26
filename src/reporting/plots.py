import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_epoch_losses(train_losses, val_losses, fname, title = None):
    """Plot training and validation loss for each epoch.""" 
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'ro-', color='blue', label='train loss over epoch')
    plt.plot(val_losses, 'ro-', color='red', label='validation loss after epoch')
    plt.xlabel('#epochs')
    plt.ylabel('avg. loss')
    plt.legend()
    if title:
        plt.title(title)
    plt.savefig(fname)
    plt.close()

def plot_epoch_accuracies(val_accuracies, fname, title = None):
    """Plot validation accuracy for each epoch.""" 
    plt.plot(val_accuracies, 'ro-', color='red', label='validation accuracy after epoch')
    plt.xlabel('#epochs')
    plt.ylabel('avg. accuracy')
    plt.legend()
    if title:
        plt.title(title)
    plt.savefig(fname)
    plt.close()

def plot_accuracy_per_position(accuracy_results, model_names, fname, title = None):
    """Plot accuracies per position in sequence.""" 
    for acc, mname in zip(accuracy_results, model_names):
        plt.plot(acc, 'o-', label=mname)
    plt.xlabel('sequence length')
    plt.ylabel('avg. accuracy')
    plt.legend()
    if title:
        plt.title(title)
    plt.savefig(fname)
    plt.close()
