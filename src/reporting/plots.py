import matplotlib
import itertools
import numpy as np
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


# Sample code from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(confusion_matrix, languages_idxs, counts, id2lang, fname, title=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(len(languages_idxs[0]), len(languages_idxs[1])))

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    languages_x = [id2lang[language_idx] for language_idx in languages_idxs[0]]
    languages_y = [id2lang[language_idx] for language_idx in languages_idxs[1]]

    tick_marks_x = np.arange(len(languages_x))
    tick_marks_y = np.arange(len(languages_y))

    plt.xticks(tick_marks_x, languages_x, rotation=45)
    plt.yticks(tick_marks_y, languages_y)

    fmt = '.5f' if not counts else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), 
                                  range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if title:
        plt.title(title)
    plt.savefig(fname)
    plt.close()
