import torch
import torch.nn as nn
import config
from src.reporting.metrics import *
from src.reporting.plots import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    PAD_index = config.settings['PAD_index']
    targets_dictionaries = torch.load(config.filepaths['targets_dictionaries'])[1]
    model_name = config.settings['model_name']
    
    (log_probs_train, targets_train, lengths) = torch.load(config.settings[model_name]['predictions_train'], device)
    (log_probs_test, targets_test, lengths) = torch.load(config.settings[model_name]['predictions_test'], device)

    nll_loss = nn.NLLLoss(ignore_index = PAD_index) # ignores target values for padding
    train_loss = calculate_loss(log_probs_train, targets_train, nll_loss, model_name)
    test_loss = calculate_loss(
        log_probs_test, targets_test, nll_loss, model_name
    )

    accuracy_test_avg  = calculate_accuracy(log_probs_test.cpu().numpy(), targets_test.cpu().numpy())
    accuracy_train_avg  = calculate_accuracy(log_probs_train.cpu().numpy(), targets_train.cpu().numpy())

    print('avg. train_loss', train_loss)
    print('avg. train_accuracy', accuracy_train_avg)

    print('avg. test_loss', test_loss)
    print('avg. test_accuracy', accuracy_test_avg)

    accuracy_test  = calculate_accuracy(log_probs_test.cpu().numpy(), targets_test.cpu().numpy(), t_axis=0)
    accuracy_train  = calculate_accuracy(log_probs_train.cpu().numpy(), targets_train.cpu().numpy(), t_axis=0)

    print(accuracy_test)

    fpath = f"test_accuracies_{model_name}.txt"
    
    with open(fpath, 'w') as f_out:
        print(accuracy_test, file=f_out)

    confusion_matrix_test, languages_idxs_test = calculate_confusion_matrix(log_probs_test.cpu().numpy(), 
                                                                            targets_test.cpu().numpy())
    confusion_matrix_train, languages_idxs_train = calculate_confusion_matrix(log_probs_train.cpu().numpy(), 
                                                                            targets_train.cpu().numpy())

    epoch_metrics = torch.load(config.settings[model_name]['epoch_metrics'])

    plot_epoch_losses(
        epoch_metrics['train_losses'], 
        epoch_metrics['val_losses'], 
        config.filepaths['plot_epoch_losses']
    )
    plot_epoch_accuracies(
        epoch_metrics['val_accuracies'], 
        config.filepaths['plot_epoch_accuracies']
    )

    plot_accuracy_per_position(
        [accuracy_test, accuracy_train],
        [f'{model_name} test', f'{model_name} train'],
        config.filepaths['plot_accuracy_seq_length']
    )

    naive_bayes_accuracies = torch.load(config.filepaths['naive_bayes_accuracies'])

    if 'word' in config.settings['model_name']:
        plot_accuracy_per_position(
            [accuracy_test, naive_bayes_accuracies['word']],
            ['RNN', 'NB'], 
            config.filepaths['plot_accuracy_model_comparison'])
    else :
        plot_accuracy_per_position(
            [ 
                accuracy_test, 
                naive_bayes_accuracies['char_word_features'],
                naive_bayes_accuracies['char_char_features']
            ],
            ['RNN', 'NB-words', 'NB-chars'], 
            config.filepaths['plot_accuracy_model_comparison']
        )


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
