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
    _ = plt.savefig(fname)
    plt.close()

def plot_accuracy_per_position(accuracies_test, accuracies_train, fname):
    """Plot accuracies per position in sequence.""" 
    plt.plot(accuracies_train, 'ro-', color='blue', label='train')
    plt.plot(accuracies_test, 'ro-', color='red', label='test')
    plt.xlabel('sequence length')
    plt.ylabel('avg. accuracy')
    plt.legend()
    plt.title('Average accuracy vs sequence length')
    _ = plt.savefig(fname)
    plt.close()
