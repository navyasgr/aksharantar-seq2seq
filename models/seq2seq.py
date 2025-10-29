import torch
import torch.nn as nn
from models.encoder import EncoderRNN
from models.decoder import DecoderRNN

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, cell_type="lstm", use_attention=False):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderRNN(input_dim, hidden_dim, cell_type)
        self.decoder = DecoderRNN(output_dim, hidden_dim, cell_type, use_attention)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_len = tgt.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, max_len, vocab_size).to(src.device)

        encoder_outputs, hidden = self.encoder(src)
        input = tgt[:, 0]

        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs
