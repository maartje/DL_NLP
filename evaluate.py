import torch
import torch.nn as nn
import config
from src.reporting.metrics import *
from src.reporting.plots import *

def main():
    PAD_index = config.settings['PAD_index']
    targets_dictionaries = torch.load(config.filepaths['targets_dictionaries'])[1]
    (log_probs_train, targets_train, lengths) = torch.load(config.filepaths['predictions_train'])
    (log_probs_test, targets_test, lengths) = torch.load(config.filepaths['predictions_test'])

    nll_loss = nn.NLLLoss(ignore_index = PAD_index) # ignores target values for padding
    train_loss = calculate_loss(log_probs_train, targets_train, nll_loss)
    test_loss = calculate_loss(log_probs_test, targets_test, nll_loss)

    accuracy_test_avg  = calculate_accuracy(log_probs_test.numpy(), targets_test.numpy())
    accuracy_train_avg  = calculate_accuracy(log_probs_train.numpy(), targets_train.numpy())

    print('avg. train_loss', train_loss)
    print('avg. train_accuracy', accuracy_train_avg)

    print('avg. test_loss', test_loss)
    print('avg. test_accuracy', accuracy_test_avg)

    accuracy_test  = calculate_accuracy(log_probs_test.numpy(), targets_test.numpy(), t_axis=0)
    accuracy_train  = calculate_accuracy(log_probs_train.numpy(), targets_train.numpy(), t_axis=0)

    confusion_matrix_test, languages_idxs_test = calculate_confusion_matrix(log_probs_test.numpy(), 
                                                                            targets_test.numpy())
    confusion_matrix_train, languages_idxs_train = calculate_confusion_matrix(log_probs_train.numpy(), 
                                                                            targets_train.numpy())

    epoch_metrics = torch.load(config.filepaths['epoch_metrics'])

    epoch_metrics = torch.load(config.filepaths['epoch_metrics'])
    
    plot_epoch_losses(
        epoch_metrics['train_losses'], 
        epoch_metrics['val_losses'], 
        config.filepaths['plot_epoch_losses']
    )
    plot_epoch_accuracies(
        epoch_metrics['val_accuracies'], 
        config.filepaths['plot_epoch_accuracies'])

    plot_accuracy_per_position(
        [accuracy_test, accuracy_train],
        ['RNN test', 'RNN train'], 
        config.filepaths['plot_accuracy_seq_length'])
        
    accuracies_test_tfidf = torch.load(config.filepaths['tf_idf_test_accuracies'])

    plot_accuracy_per_position(
        [accuracy_test, accuracies_test_tfidf],
        ['RNN', 'TFIDF'], 
        config.filepaths['plot_accuracy_model_comparison'])

    plot_confusion_matrix(
        confusion_matrix_test,
        languages_idxs_test,
        False,
        targets_dictionaries,
        config.filepaths['plot_test_confusion_matrix']
    )

    plot_confusion_matrix(
        confusion_matrix_train,
        languages_idxs_train,
        False,
        targets_dictionaries,
        config.filepaths['plot_train_confusion_matrix']
    )

if __name__ == "__main__":
    main()
