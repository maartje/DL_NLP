import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import tables
from torch.utils import data
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DatasetLanguageIdentification(data.Dataset):

    def __init__(self, fpath_vectors, fpath_labels, max_seq_length):
        super(DatasetLanguageIdentification, self).__init__()
        
        self.sequence_vectors = torch.load(fpath_vectors, device) 
        self.labels = torch.load(fpath_labels, device) # TODO: read pickle files        
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.sequence_vectors)

    def __getitem__(self, index):
        input_seq = self.sequence_vectors[index][:self.max_seq_length]
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target = torch.tensor(self.labels[index], dtype=torch.long)
        return input_seq, target

def collate_seq_vectors(batch, PAD_index, check_for_equal_length = True):   
    transposed = list(zip(*batch))

    # pad, sort and stack
    sequences = transposed[0]
    sequence_lengths = np.array([len(c) for c in sequences])
    sort_indices = np.argsort(sequence_lengths)[::-1].copy()
    max_length = max(sequence_lengths)

    if check_for_equal_length:
        assert np.all(np.equal(sequence_lengths,max_length)), "Sequences of different length are not fully support (yet)"
    
    sequences_padded = [
        torch.cat(
            (c, torch.LongTensor([PAD_index] * (max_length - len(c))))
        )  for c in sequences]   
    sequences_collated = default_collate(sequences_padded)
    sequences_collated_sorted = sequences_collated[sort_indices]
    
    targets = transposed[1]
    targets = [(sp != PAD_index).to(dtype=torch.long) * t.item() for sp, t in zip(sequences_padded, targets)]
    targets_collated_sorted = default_collate(targets)[sort_indices]

    return [
        sequences_collated_sorted, 
        targets_collated_sorted, 
        torch.tensor(sequence_lengths[sort_indices], dtype=torch.long)
    ]

