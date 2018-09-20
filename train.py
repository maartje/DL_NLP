from src.io.dataset_language_identification import DatasetLanguageIdentification, collate_seq_vectors
from torch.utils import data
import config

def main():
    fpath_vectors_train = config.filepaths['vectors_train'] 
    fpath_labels_train = config.filepaths['labels_train']
    PAD_index = config.settings['PAD_index']
    batch_size = config.settings['batch_size']

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


if __name__ == "__main__":
    main()
