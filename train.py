from src.io.dataset_language_identification import DatasetLanguageIdentification, collate_seq_vectors
from torch.utils import data
import config
from src.nn.rnn_model import LanguageRecognitionRNN
from src.nn.train import fit
import torch.nn as nn
import torch.optim as optim

def main():
    fpath_vectors_train = config.filepaths['vectors_train'] 
    fpath_labels_train = config.filepaths['targets_train']
    PAD_index = config.settings['PAD_index']
    batch_size = config.settings['rnn']['batch_size']

    ds_train = DatasetLanguageIdentification(
        fpath_vectors_train, 
        fpath_labels_train
    )

    dl_params_train = {
        'batch_size' : batch_size,
        'collate_fn' : lambda b: collate_seq_vectors(b, PAD_index),
        'shuffle' : True
    }

    dl_train = data.DataLoader(ds_train, **dl_params_train)

    emb_size = config.settings['rnn']['emb_size'] #64
    hidden_size = config.settings['rnn']['hidden_size'] #256
    output_size = 3 # nr of languages + 1 for padding (pass as a parameter read from label dict)
    drop_out = config.settings['rnn']['drop_out'] #0.3
    model = LanguageRecognitionRNN(
        emb_size, hidden_size, output_size, PAD_index, drop_out)

    learning_rate = config.settings['rnn']['learning_rate'] #0.1
    loss = nn.NLLLoss(ignore_index = PAD_index) # ignores a target value
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)

    epochs = config.settings['rnn']['epochs'] #250
    fit(model, dl_train, loss, optimizer, epochs, [])


if __name__ == "__main__":
    main()
