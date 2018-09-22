from src.io.dataset_language_identification import DatasetLanguageIdentification, collate_seq_vectors
from torch.utils import data
import config
from src.nn.rnn_model import LanguageRecognitionRNN
from src.nn.train import fit
import torch.nn as nn
import torch.optim as optim
from src.reporting.loss_collector import LossCollector
from src.reporting.train_output_writer import TrainOutputWriter
import torch

def main():
    fpath_vectors_train = config.filepaths['vectors_train'] 
    fpath_labels_train = config.filepaths['targets_train']
    PAD_index = config.settings['PAD_index']
    batch_size = config.settings['rnn']['batch_size']

    # initialize data loader
    ds_train = DatasetLanguageIdentification(
        fpath_vectors_train, 
        fpath_labels_train,
        config.settings['max_seq_length']
    )
    dl_params_train = {
        'batch_size' : batch_size,
        'collate_fn' : lambda b: collate_seq_vectors(b, PAD_index),
        'shuffle' : True
    }
    dl_train = data.DataLoader(ds_train, **dl_params_train)

    # initialize RNN model and train settings
    emb_size = config.settings['rnn']['emb_size'] 
    hidden_size = config.settings['rnn']['hidden_size'] 
    output_size = 3 # nr of languages + 1 for padding (pass as a parameter read from label dict)
    drop_out = config.settings['rnn']['drop_out'] 
    model = LanguageRecognitionRNN(
        emb_size, hidden_size, output_size, PAD_index, drop_out)

    # initialize train settings for RNN model
    learning_rate = config.settings['rnn']['learning_rate'] 
    loss = nn.NLLLoss(ignore_index = PAD_index) # ignores target value 0
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    epochs = config.settings['rnn']['epochs'] 

    # collect information during training
    lossCollector = LossCollector()
    trainOutputWriter = TrainOutputWriter(lossCollector)

    # fit RNN model
    fit(model, dl_train, loss, optimizer, epochs, [
        lossCollector.store_train_loss,
        trainOutputWriter.print_epoch_info
    ])

    torch.save({
        'train_losses' : lossCollector.train_losses,
        'val_losses' : lossCollector.val_losses # TODO
    }, config.filepaths['epoch_losses'])


if __name__ == "__main__":
    main()
