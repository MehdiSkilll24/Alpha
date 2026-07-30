import torch

MAX_LEN = 1024 

def collate_fn(batch, pad_id=0):
    src_batch = []
    target_batch = []

    for src, tgt in batch:
        src_batch.append(src[:MAX_LEN])
        target_batch.append(tgt[:MAX_LEN])

    max_src = max(len(sentence) for sentence in src_batch)
    max_tgt = max(len(sentence) for sentence in target_batch)

    src_batch = [
        sentence + [pad_id] * (max_src - len(sentence))
        for sentence in src_batch
    ]

    tgt_batch = [
        sentence + [pad_id] * (max_tgt - len(sentence))
        for sentence in target_batch
    ]

    src_batch = torch.tensor(src_batch)
    tgt_batch = torch.tensor(tgt_batch)

    return src_batch, tgt_batch