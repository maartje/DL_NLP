import torch
import torch.nn as nn
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class LanguageRecognitionCNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 output_size,
                 pad_index,
                 drop_out,
                 n_kernels=100,
                 kernel_sizes=[3,4,5]):

        super(LanguageRecognitionCNN, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.kernel_sizes = kernel_sizes
        self.embedding_size = embedding_size
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.convs = nn.ModuleList(
            [nn.Conv2d(
                1, n_kernels, 
                [k, self.embedding_size], 
                padding = (k - 1, 0)
            ) 
            for k in self.kernel_sizes]
        )

        self.linear = nn.Linear(len(self.kernel_sizes) * n_kernels, self.output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1) # TODO: or cross entropy loss?
        
        # self.dropout_embedding = nn.Dropout(p=drop_out)
        self.dropout_cnn = nn.Dropout(p=drop_out)

    def forward(self, input_data, seq_lengths):
        x = self.embedding(input_data)
       
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(-1) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        x = x.view(x.size(0), -1)
        
        output = self.linear(self.dropout_cnn(x))  # unpacked[0]))
        output = self.logsoftmax(output) 

        return output #, hidden
