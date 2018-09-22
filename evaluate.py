import torch
import torch.nn as nn
import config

def main():
    PAD_index = config.settings['PAD_index']
    (log_probs_train, targets_train, _) = torch.load(config.filepaths['predictions_train'])
    (log_probs_test, targets_test, _) = torch.load(config.filepaths['predictions_test'])
    train_loss = calculate_loss(log_probs_train, targets_train, PAD_index)
    test_loss = calculate_loss(log_probs_test, targets_test, PAD_index)


    print('train_loss', train_loss)
    print('test_loss', test_loss)

def calculate_loss(log_probs, targets, PAD_index):
    nll_loss = nn.NLLLoss(ignore_index = PAD_index) # ignores target values for padding
    loss = nll_loss(log_probs.permute(0,2,1), targets)
    # TODO might be insightfull to show average loss per string position
    # this might be possible using "reduction = 'none'" and some clever
    # reshaping so that we do reduce over batches but not over positions 
    return loss

if __name__ == "__main__":
    main()
