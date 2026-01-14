import torch
import torch.nn as nn

# -------------------- Encoder --------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return hidden

# -------------------- Decoder --------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

# -------------------- Seq2Seq --------------------
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size)

    def forward(self, src, trg):
        # Encode source
        hidden = self.encoder(src)

        # Decode target
        outputs, _ = self.decoder(trg, hidden)
        return outputs

    # Greedy decoding for inference
    def greedy_decode(self, src, max_len=30, start_idx=1, end_idx=2):
        self.eval()
        with torch.no_grad():
            hidden = self.encoder(src)
            batch_size = src.size(0)
            inputs = torch.tensor([[start_idx]] * batch_size, device=src.device)

            outputs = []
            for _ in range(max_len):
                out, hidden = self.decoder(inputs, hidden)
                pred = out.argmax(2)  # batch x 1
                outputs.append(pred)
                inputs = pred
            outputs = torch.cat(outputs, dim=1)  # batch x max_len
        return outputs
