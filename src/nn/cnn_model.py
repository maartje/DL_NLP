import torch
import torch.nn as nn
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class LanguageRecognitionCNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 output_size,
                 pad_index,
                 drop_out,
                 n_kernels=100,
                 kernel_sizes=[3,4,5],
                 embedding_size=128):

        super(LanguageRecognitionCNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pad_index = pad_index
        self.kernel_sizes = kernel_sizes
        self.embedding_size = embedding_size
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, n_kernels, (k, self.embedding_size)) 
                                    for k in self.kernel_sizes])

        self.out = nn.Linear(len(self.kernel_sizes), self.output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2) # TODO: or cross entropy loss?
        
        self.dropout_embedding = nn.Dropout(p = drop_out)
        self.dropout_cnn = nn.Dropout(p = drop_out)

    def forward(self, input_data, seq_lengths):
        output = self.dropout_embedding(self.embedding(input_data))
        # packed = pack_padded_sequence (
        #     output, seq_lengths, batch_first=True)
        # output, hidden = self.lstm(packed)
        # unpacked = pad_packed_sequence(
        #     output, batch_first=True, padding_value=self.pad_index, total_length=None)
        x = output.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        output = self.out(self.dropout_cnn(x)) #unpacked[0]))
        output = self.logsoftmax(output) 

        return output #, hidden
