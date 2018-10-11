import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data.dataset import random_split

import config
from src.io.dataset_language_identification import (DatasetLanguageIdentification,
                                                    collate_seq_vectors)
from src.model_saver import ModelSaver
from src.nn.cnn_model import LanguageRecognitionCNN
from src.nn.rnn_model import LanguageRecognitionRNN
from src.nn.train import fit
from src.reporting.metrics_collector import MetricsCollector
from src.reporting.train_output_writer import TrainOutputWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    fpath_vectors_train = config.filepaths['vectors_train']
    fpath_labels_train = config.filepaths['targets_train']
    PAD_index = config.settings['PAD_index']
    model_name = config.settings['model_name']
    batch_size = config.settings[model_name]['batch_size']

    print("model: ", config.settings['model'])
    print("model name: ", model_name)
    print("batch size: ", batch_size)
    print("max length: ", config.settings['max_seq_length'])
    print("learning_rate", config.settings[model_name]['learning_rate'])
    print("number of classes: ", len(config.language_filters['test']))

    # initialize data loader
    ds = DatasetLanguageIdentification(
        fpath_vectors_train,
        fpath_labels_train,
        config.settings['max_seq_length']
    )

    dl_params_train = {
        'batch_size': batch_size,
        'collate_fn': lambda b: collate_seq_vectors(b, PAD_index, False),
        'pin_memory': True if torch.cuda.is_available() else False,
        'shuffle': True
    }
    dl_params_val = {
        'batch_size': batch_size,  # or setting?
        'collate_fn': lambda b: collate_seq_vectors(b, PAD_index, False),
        'pin_memory': True if torch.cuda.is_available() else False,
        'shuffle': False
    }

    val_size = math.ceil(config.settings['val_train_ratio'] * len(ds))
    train_size = len(ds) - val_size

    ds_train, ds_val = random_split(
        ds, [train_size, val_size])

    dl_train = data.DataLoader(ds_train, **dl_params_train)
    dl_val = data.DataLoader(ds_val, **dl_params_val)

    # initialize the model and train settings
    vocab_size = torch.load(config.filepaths['vocab'], device).vocab.n_words
    hidden_size = config.settings[model_name]['hidden_size']
    # nr of languages + 1 for padding (pass as a parameter read from label dict)
    output_size = len(torch.load(config.filepaths['targets_dictionaries'], device)[0])
    drop_out = config.settings[model_name]['drop_out']

    if model_name == 'rnn':
        model = LanguageRecognitionRNN(
            vocab_size, hidden_size, output_size, PAD_index, drop_out)
    if model_name == 'cnn':
        model = LanguageRecognitionCNN(vocab_size, hidden_size, output_size, PAD_index, drop_out)
    
    # model = torch.load('data/train/model.pt')

    model = nn.DataParallel(model).to(device)
    
    # initialize train settings for the model
    learning_rate = config.settings[model_name]['learning_rate']

    loss = nn.NLLLoss(ignore_index=PAD_index)  # ignores target value 0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = config.settings[model_name]['epochs']

    # gitoptimizer = optim.SGD(model.parameters(), lr = learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr = 0.001)

    # collect information during training
    metricsCollector = MetricsCollector(
        model, dl_val, config.settings['max_seq_length'], loss, model_name
    )
    trainOutputWriter = TrainOutputWriter(metricsCollector)

    modelSaver = ModelSaver(model, metricsCollector, config.filepaths['model'])

    # fit the model
    fit(model, dl_train, loss, optimizer, epochs, model_name, torch.device(device), [
        metricsCollector.store_metrics,
        trainOutputWriter.print_epoch_info,
        modelSaver.save_best_model
    ])

    # save metrics collected during training
    torch.save({
        'train_losses': metricsCollector.train_losses,
        'val_losses': metricsCollector.val_losses,
        'val_accuracies': metricsCollector.val_accuracies
    }, config.filepaths['epoch_metrics'])


if __name__ == "__main__":
    main()
