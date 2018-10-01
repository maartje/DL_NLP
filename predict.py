import torch
import config
from src.nn.train import predict
from src.io.dataset_language_identification import DatasetLanguageIdentification, collate_seq_vectors
from torch.utils import data

def build_dataloader(fpath_vectors, fpath_labels):
    PAD_index = config.settings['PAD_index']
    batch_size = config.settings['rnn']['batch_size']
    ds = DatasetLanguageIdentification(
        fpath_vectors, 
        fpath_labels,
        config.settings['max_seq_length']
    )
    dl_params = {
        'batch_size' : batch_size,
        'collate_fn' : lambda b: collate_seq_vectors(b, PAD_index, config.settings['check_equal_seq_length']),
        'shuffle' : False
    }
    return data.DataLoader(ds, **dl_params)

def main():
    dl_test = build_dataloader(
        config.filepaths['vectors_test'], 
        config.filepaths['targets_test']
    )
    dl_train = build_dataloader(
        config.filepaths['vectors_train'], 
        config.filepaths['targets_train']
    )
    model = torch.load(config.filepaths['model'])

    test_results = predict(model, dl_test, config.settings['max_seq_length'])
    train_results = predict(model, dl_train, config.settings['max_seq_length'])

    torch.save(test_results, config.filepaths['predictions_test'])
    torch.save(train_results, config.filepaths['predictions_train'])



if __name__ == "__main__":
    main()
