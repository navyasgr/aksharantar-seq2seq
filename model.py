import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Encoder
# ============================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, cell_type="lstm"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)

        if cell_type == "gru":
            self.rnn = nn.GRU(hidden_dim, hidden_dim)
        else:
            self.rnn = nn.LSTM(hidden_dim, hidden_dim)

        self.cell_type = cell_type

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        if self.cell_type == "lstm":
            hidden, cell = hidden
            return outputs, (hidden, cell)
        else:
            return outputs, (hidden, None)


# ============================================================
# Attention
# ============================================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(torch.sum(energy, dim=2)).T
        return F.softmax(attention, dim=1)


# ============================================================
# Decoder
# ============================================================
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, cell_type="lstm", attention=False):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.attention_enabled = attention
        self.cell_type = cell_type

        if attention:
            self.attention = Attention(hidden_dim)
            self.rnn = nn.LSTM(hidden_dim * 2, hidden_dim)
        else:
            self.rnn = nn.LSTM(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs=None):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)

        if self.attention_enabled and encoder_outputs is not None:
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            attn_weights = attn_weights.unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_outputs.permute(1, 0, 2))
            rnn_input = torch.cat((embedded, context.permute(1, 0, 2)), dim=2)
        else:
            rnn_input = embedded

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


# ============================================================
# Seq2Seq Model
# ============================================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        input = tgt[0, :]

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = tgt[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs


# ============================================================
# Builder
# ============================================================
def build_model(input_dim, output_dim, hidden_dim, device, cell_type="lstm", attention=False):
    encoder = Encoder(input_dim, hidden_dim, cell_type)
    decoder = Decoder(output_dim, hidden_dim, cell_type, attention)
    model = Seq2Seq(encoder, decoder, device).to(device)
    return model
