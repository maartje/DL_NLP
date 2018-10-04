import torch
import config
from src.nn.train import predict
from src.io.dataset_language_identification import DatasetLanguageIdentification, collate_seq_vectors
from torch.utils import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_dataloader(fpath_vectors, fpath_labels):
    model_name = config.settings['model_name']
    PAD_index = config.settings['PAD_index']
    batch_size = config.settings[model_name]['batch_size']
    ds = DatasetLanguageIdentification(
        fpath_vectors,
        fpath_labels,
        config.settings[model_name]['max_seq_length']
    )
    dl_params = {
        'batch_size' : batch_size,
        'collate_fn' : lambda b: collate_seq_vectors(b, PAD_index, config.settings['check_equal_seq_length']),
        'pin_memory': True if torch.cuda.is_available() else False,
        'shuffle' : False
    }
    return data.DataLoader(ds, **dl_params)

def main():
    model_name = config.settings['model_name']
    dl_test = build_dataloader(
        config.filepaths['vectors_test'],
        config.filepaths['targets_test']
    )
    dl_train = build_dataloader(
        config.filepaths['vectors_train'],
        config.filepaths['targets_train']
    )
    model = torch.load(config.settings[model_name]['model_path'], device)
    
    if type(model) == torch.nn.DataParallel:
        model = model.module

    test_results = predict(
        model, dl_test, config.settings[model_name]['max_seq_length'],
        model_name, device
    )
    train_results = predict(
        model, dl_train, config.settings[model_name]['max_seq_length'], 
        model_name, device
    )

    torch.save(test_results, config.settings[model_name]['predictions_test'])
    torch.save(train_results, config.settings[model_name]['predictions_train'])



if __name__ == "__main__":
    main()
