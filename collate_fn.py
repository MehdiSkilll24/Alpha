import torch
import functools

MAX_LEN = 1024

def collate_fn(pad_id):
    """Returns a collate function with the correct pad_id baked in."""
    @functools.wraps(collate_fn)
    
    def _collate(batch):
        batch = [tokens[:MAX_LEN] for tokens in batch]
        max_len = max(len(sentence) for sentence in batch)
        batch = [tokens + [pad_id] * (max_len - len(tokens)) for tokens in batch]
        return torch.tensor(batch, dtype=torch.long)
    return _collate