import torch

MAX_LEN = 1024  # keeps RoPE's max_seq_len=1024 buffer safely sufficient, prevents outlier-driven OOM

def collate_fn(batch, pad_id=0):

    batch = [
        tokens[:MAX_LEN]
        for tokens in batch
    ]

    
    max_len = max(len(sentence) for sentence in batch)

    batch = [
        tokens + [pad_id] * (max_len - len(tokens))
        for tokens in batch
    ]

    return torch.tensor(batch, dtype=torch.long)