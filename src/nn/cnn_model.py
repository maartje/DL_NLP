import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LanguageRecognitionCNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, pad_index, drop_out):
        super(LanguageRecognitionCNN, self).__init__()
        pass

    def forward(self, input_data, seq_lengths):
        pass
