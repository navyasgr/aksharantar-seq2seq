import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, timestep, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = torch.matmul(energy, self.v)
        attn_weights = F.softmax(energy, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context, attn_weights

class DecoderRNN(nn.Module):
    def __init__(self, output_dim, hidden_dim, cell_type="lstm", use_attention=False):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.lower()
        self.use_attention = use_attention

        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.attention = BahdanauAttention(hidden_dim) if use_attention else None

        if self.cell_type == "gru":
            self.rnn = nn.GRU(hidden_dim * (2 if use_attention else 1), hidden_dim, batch_first=True)
        elif self.cell_type == "rnn":
            self.rnn = nn.RNN(hidden_dim * (2 if use_attention else 1), hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(hidden_dim * (2 if use_attention else 1), hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)

        if self.use_attention:
            if isinstance(hidden, tuple):
                h = hidden[0][-1]
            else:
                h = hidden[-1]
            context, _ = self.attention(h, encoder_outputs)
            rnn_input = torch.cat((embedded, context), dim=2)
        else:
            rnn_input = embedded

        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden
