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

    # TODO accuracy_test  = calculate_accuracies(log_probs_test, targets_test, lengths)
    # TODO accuracy_train = calculate_accuracies(log_probs_test, targets_test, lengths)

    print('train_loss', train_loss)
    print('test_loss', test_loss)

    epoch_metrics = torch.load(config.filepaths['epoch_metrics'])
    print('epoch metrics', epoch_metrics)
    # TODO: plot the metrics to file

    epoch_metrics = torch.load(config.filepaths['epoch_metrics'])
    plot_epoch_losses(
        epoch_metrics['train_losses'], 
        epoch_metrics['val_losses'], 
        config.filepaths['plot_epoch_losses']
    )

if __name__ == "__main__":
    main()
