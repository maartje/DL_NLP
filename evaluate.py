import torch
import torch.nn as nn
import config
from src.reporting.metrics import *
from src.reporting.plots import *

def main():
    PAD_index = config.settings['PAD_index']
    (log_probs_train, targets_train, lengths) = torch.load(config.filepaths['predictions_train'])
    (log_probs_test, targets_test, lengths) = torch.load(config.filepaths['predictions_test'])

    nll_loss = nn.NLLLoss(ignore_index = PAD_index) # ignores target values for padding
    train_loss = calculate_loss(log_probs_train, targets_train, nll_loss)
    test_loss = calculate_loss(log_probs_test, targets_test, nll_loss)

    accuracy_test  = calculate_accuracy(log_probs_test.numpy(), targets_test.numpy(), t_axis=0)
    accuracy_train  = calculate_accuracy(log_probs_train.numpy(), targets_train.numpy(), t_axis=0)

    print('avg. train_loss', train_loss)
    print('avg. train_accuracy', accuracy_train.mean())

    print('avg. test_loss', test_loss)
    print('avg. test_accuracy', accuracy_test.mean())

    epoch_metrics = torch.load(config.filepaths['epoch_metrics'])

    epoch_metrics = torch.load(config.filepaths['epoch_metrics'])
    plot_epoch_losses(
        epoch_metrics['train_losses'], 
        epoch_metrics['val_losses'], 
        config.filepaths['plot_epoch_losses']
    )
    plot_epoch_accuracies(epoch_metrics['val_accuracies'], config.filepaths['plot_epoch_accuracies'])

    plot_accuracy_per_position(
        accuracy_test, 
        accuracy_train, 
        config.filepaths['plot_accuracy_seq_length'])

if __name__ == "__main__":
    main()
