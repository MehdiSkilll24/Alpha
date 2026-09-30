import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PackedBinDataset(Dataset):
    def __init__(self, path, seq_len=1024):
        self.path = path
        self.seq_len = seq_len
        n_tokens = os.path.getsize(path) // 2          # uint16 = 2 bytes
        self.num_sequences = (n_tokens - 1) // seq_len
        self.data = None                               # opened lazily, per worker

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self.data is None:
            self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return torch.from_numpy(chunk.astype(np.int64))