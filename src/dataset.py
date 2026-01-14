import torch
from torch.utils.data import Dataset

class CodeSummaryDataset(Dataset):
    def __init__(self, data, code_vocab, summary_vocab, max_code_len, max_sum_len):
        self.data = data
        self.code_vocab = code_vocab
        self.summary_vocab = summary_vocab
        self.max_code_len = max_code_len
        self.max_sum_len = max_sum_len

    def pad(self, seq, max_len):
        return seq[:max_len] + [0] * (max_len - len(seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code, summary = self.data[idx]

        code_ids = self.code_vocab.numericalize(code)
        summary_ids = [1] + self.summary_vocab.numericalize(summary) + [2]

        code_ids = self.pad(code_ids, self.max_code_len)
        summary_ids = self.pad(summary_ids, self.max_sum_len)

        return torch.tensor(code_ids), torch.tensor(summary_ids)
