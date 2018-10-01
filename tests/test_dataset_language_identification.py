"""
Tests for the language identification dataset
"""

import unittest
import mock
from torch.utils import data
import numpy as np
from src.io.dataset_language_identification import DatasetLanguageIdentification, collate_seq_vectors

fpath_vectors = 'data/preprocess/vectors_train.pt'
fpath_labels = 'data/preprocess/labels_train.pt'

def mock_torch_load(fpath):
    if fpath == fpath_vectors:
        return [
            [3,2,7,5,1], # the indices of the tokens in the sentence
            [8,2,2,7,9,9,6,5,7,6,4,1,3,3,4],
            [7,8,2,5,6,6,7,1],
            [8,5,2,7,9,4,6,5,6,2,1]
        ]
    if fpath == fpath_labels:
        return [
            4, # the index of the language for the sentence
            1,
            3,
            2
        ]

class TestDatasetLanguageIdentification(unittest.TestCase):
    
    @mock.patch('torch.load', side_effect = mock_torch_load)
    def setUp(self, torch_load):
        self.ds_lang_id = DatasetLanguageIdentification(
            fpath_vectors, 
            fpath_labels,
            12
        )
        PAD_index = 0
        collate_fn = lambda b: collate_seq_vectors(b, PAD_index, False)

        dl_params = {
            'batch_size' : 3,
            'collate_fn' : collate_fn,
            'shuffle' : False
        }
        self.dl_lang_id = data.DataLoader(self.ds_lang_id, **dl_params)

    def test_collate_seq_vectors(self):
        batch = next(iter(self.dl_lang_id))

        expected_sequences = [ # sorted and padded
            [8,2,2,7,9,9,6,5,7,6,4,1],
            [7,8,2,5,6,6,7,1,0,0,0,0],
            [3,2,7,5,1,0,0,0,0,0,0,0]
        ]
        expected_targets = [ # targets for each character, sorted and padded
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [3,3,3,3,3,3,3,3,0,0,0,0],
            [4,4,4,4,4,0,0,0,0,0,0,0]
        ]
        expected_seq_lengths = [12,8,5]
        self.assertTrue(np.array_equal(batch[0], expected_sequences))
        self.assertTrue(np.array_equal(batch[1], expected_targets))
        self.assertTrue(np.array_equal(batch[2], expected_seq_lengths))

if __name__ == '__main__':
    unittest.main()


