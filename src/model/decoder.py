import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, attention, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = attention
        self.lstm = nn.LSTM(
            embed_size + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)

        context, attn = self.attention(hidden, encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)

        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))

        return predictions, hidden, cell
