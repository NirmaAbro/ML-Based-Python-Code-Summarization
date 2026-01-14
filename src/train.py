# src/train.py
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from vocab import Vocabulary
from model.seq2seq import Seq2Seq

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------- Hyperparameters --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 5
LR = 0.001
MAX_CODE_LEN = 200
MAX_SUM_LEN = 30

# -------------------- Dataset --------------------
class CodeSummaryDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code, summary = self.data[idx]
        src_ids = self.vocab.numericalize(code)
        trg_ids = self.vocab.numericalize(summary)
        return torch.tensor(src_ids), torch.tensor(trg_ids)

# -------------------- Data Loading --------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------- Training Functions --------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, trg in tqdm(loader):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:,1:].reshape(-1, output_dim)
        trg = trg[:,1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[:,1:].reshape(-1, output_dim)
            trg = trg[:,1:].reshape(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()
    return total_loss / len(loader)

# -------------------- Main --------------------
def main():
    # Load preprocessed data
    train_data = load_json(os.path.join(DATA_DIR, "train.json"))
    val_data = load_json(os.path.join(DATA_DIR, "validation.json"))

    print(f"Train samples: {len(train_data)} | Validation samples: {len(val_data)}")

    # Build vocab from training data
    vocab = Vocabulary()
    for code, summary in train_data:
        vocab.add_sentence(code)
        vocab.add_sentence(summary)
    print(f"Vocabulary size: {len(vocab)}")
    vocab.save(os.path.join(DATA_DIR, "vocab.pkl"))

    # Datasets & loaders
    train_dataset = CodeSummaryDataset(train_data, vocab)
    val_dataset = CodeSummaryDataset(val_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: collate_fn(x))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x))

    # Build model
    model = Seq2Seq(len(vocab), EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        # Save model
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))

# -------------------- Collate Function --------------------
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    # Pad sequences
    src_lens = [len(x) for x in src_batch]
    trg_lens = [len(x) for x in trg_batch]
    max_src = max(src_lens)
    max_trg = max(trg_lens)
    padded_src = torch.zeros(len(batch), max_src, dtype=torch.long)
    padded_trg = torch.zeros(len(batch), max_trg, dtype=torch.long)
    for i, (src, trg) in enumerate(batch):
        padded_src[i, :len(src)] = src
        padded_trg[i, :len(trg)] = trg
    return padded_src, padded_trg

if __name__ == "__main__":
    main()
