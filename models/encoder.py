import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, cell_type="lstm"):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.lower()

        self.embedding = nn.Embedding(input_dim, hidden_dim)

        if self.cell_type == "gru":
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        elif self.cell_type == "rnn":
            self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden
