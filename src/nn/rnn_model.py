import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LanguageRecognitionRNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index, drop_out):
        super(LanguageRecognitionRNN, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pad_index = pad_index

        self.embedding = nn.Embedding(self.emb_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2) # TODO: or cross entropy loss?
        
        self.dropout_embedding = nn.Dropout(p = drop_out)
        self.dropout_lstm = nn.Dropout(p = drop_out)

    def forward(self, input_data, seq_lengths):
        output = self.dropout_embedding(self.embedding(input_data))
        packed = pack_padded_sequence (
            output, seq_lengths, batch_first=True)
        output, hidden = self.lstm(packed)
        unpacked = pad_packed_sequence(
            output, batch_first=True, padding_value=self.pad_index, total_length=None)
        output = self.out(self.dropout_lstm(unpacked[0]))
        output = self.logsoftmax(output) 
        return output #, hidden
