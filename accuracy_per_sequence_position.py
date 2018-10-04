import preprocess
import config
import torch
from src.nn.train import predict
from src.reporting.metrics import calculate_accuracy
from src.io.dataset_language_identification import DatasetLanguageIdentification, collate_seq_vectors
from torch.utils import data
from src.preprocess.tokenizer import CharacterTokenizer, WordTokenizer

def build_dataloader(fpath_vectors, fpath_labels, n):
    model_name = config.settings['model_name']
    PAD_index = config.settings['PAD_index']
    batch_size = config.settings[model_name]['batch_size']
    ds = DatasetLanguageIdentification(
        fpath_vectors, 
        fpath_labels,
        n
    )
    dl_params = {
        'batch_size' : batch_size,
        'collate_fn' : lambda b: collate_seq_vectors(b, PAD_index, config.settings['check_equal_seq_length']),
        'shuffle' : False
    }
    return data.DataLoader(ds, **dl_params)

def main():
    print('Assumptions: vocabulary, target_vectors, trained model')

    model_name = config.settings['model_name']

    results = []
    for n in range(1, 51):
        preprocess.preprocess_texts_max_length(n, CharacterTokenizer())

        dl_test = build_dataloader(
            config.settings[model_name]['vectors_test'], 
            config.settings[model_name]['targets_test'],
            n
        )
        model = torch.load(config.filepaths['model'])

        (log_probs_test, targets_test, _) = predict(model, dl_test, n, model_name)
        accuracy_test  = calculate_accuracy(log_probs_test.numpy(), targets_test.numpy(), t_axis=0)
        results.append(accuracy_test[-1])
        print(f"position {n}: {accuracy_test[-1]}")

    print('accuracies per sequence length', results)
    fpath = f"test_accuracies_by_looping_over_positions.txt"
    
    with open(fpath, 'w') as f_out:
        print(results, file=f_out)


if __name__ == "__main__":
    main()
