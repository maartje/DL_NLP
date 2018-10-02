import preprocess
import train_hp
import predict
import tf_idf_baseline
import torch
import torch.nn as nn
import config
from src.reporting.metrics import *
from src.reporting.plots import *


def evaluate_hp():
    PAD_index = config.settings['PAD_index']
    targets_dictionaries = torch.load(config.filepaths['targets_dictionaries'])[1]
    (log_probs_train, targets_train, lengths) = torch.load(config.filepaths['predictions_train'])
    (log_probs_test, targets_test, lengths) = torch.load(config.filepaths['predictions_test'])

    nll_loss = nn.NLLLoss(ignore_index = PAD_index) # ignores target values for padding
    train_loss = calculate_loss(log_probs_train, targets_train, nll_loss)
    test_loss = calculate_loss(log_probs_test, targets_test, nll_loss)

    accuracy_test_avg  = calculate_accuracy(log_probs_test.numpy(), targets_test.numpy())
    accuracy_train_avg  = calculate_accuracy(log_probs_train.numpy(), targets_train.numpy())




def main():
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7]

    preprocess.main()
    tf_idf_baseline.main()

    val_acc_hp = {}
    val_loss_hp = {}

    for lr in learning_rates:
    	train_hp.main(lr)
    	predict.main()
    	evaluate_hp()
    	epoch_metrics = torch.load(config.filepaths['epoch_metrics'])
    	val_acc_hp[lr] = epoch_metrics['val_accuracies'][-1]
    	val_loss_hp[lr] = epoch_metrics['val_losses'][-1]

        

    plt.plot(val_acc_hp.keys(), val_acc_hp.values(), 'ro-', color='blue', label='validation accuracy')
    plt.plot(val_loss_hp.keys(), val_loss_hp.values(), 'ro-', color='red', label='validation loss')

    plt.xlabel('learning rates')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title("hyper params")
    plt.savefig('hyper_parameters.png')
    plt.close()

if __name__ == "__main__":
    main()

