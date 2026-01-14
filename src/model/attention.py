import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [1, batch, hidden]
        # encoder_outputs: [batch, seq_len, hidden]

        decoder_hidden = decoder_hidden[-1].unsqueeze(2)
        scores = torch.bmm(encoder_outputs, decoder_hidden).squeeze(2)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context, attn_weights
