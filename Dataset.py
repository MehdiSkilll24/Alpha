from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class PackedWikiTextDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len=1024):
        self.seq_len = seq_len

        all_tokens = []

        for sample in dataset:
            text = sample["text"]

            if not text.strip():
                continue

            tokens = tokenizer.encode(text)

            # Optional: separate documents/rows with EOS
            tokens.append(tokenizer.word_to_idx["<eos>"])

            all_tokens.extend(tokens)

        # Drop incomplete final chunk
        usable_length = (len(all_tokens) // seq_len) * seq_len
        all_tokens = all_tokens[:usable_length]

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

        self.num_sequences = usable_length // seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        return self.tokens[start:end]