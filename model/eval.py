"""Evaluate the Alpha BASE model: held-out perplexity + zero-shot HellaSwag / ARC-Easy.

Usage:
    python eval.py --ckpt checkpoints/checkpoint_best.pt --val val.bin
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from bpe import load_bpe_tokenizer
from main import Transformer

SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT = SCRIPT_DIR.parent                        # .../Alpha
TOK_PATH = ROOT / "jsons" / "fineweb_bpe.json"
MAX_CTX = 2048  # RoPE table size; positions beyond this crash / are untrained

# Load the tokenizer ONCE (reloading the JSON per call makes the MC eval crawl).
TOK = load_bpe_tokenizer(TOK_PATH)
BOS = TOK.word_to_idx["<bos>"]


# ---------------------------------------------------------------- ADAPTER
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt["arch"]  # trust the checkpoint over config.json
    assert arch["vocab_size"] == len(TOK), "Tokenizer/checkpoint vocab mismatch"
    model = Transformer(
        arch["d_model"], arch["num_heads"], arch["vocab_size"],
        arch["d_ff"], arch["num_kv_heads"], arch["num_decoder_layers"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {ckpt_path} | step {ckpt.get('step')} | "
          f"val loss {ckpt.get('val_loss', ckpt.get('best_val'))}")
    return model.to(device).eval()


def encode(text, bos=False):
    """Raw BPE ids. TOK.encode() adds <bos>/<eos>, which would corrupt scoring."""
    ids = TOK.tok.encode(text, add_special_tokens=False).ids
    return [BOS] + ids if bos else ids


def logits_of(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


# ------------------------------------------------------------- PERPLEXITY
@torch.no_grad()
def perplexity(model, path, block, device, bs=16):
    """Non-overlapping windows over val.bin. Returns (ppl, mean_nll)."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n_windows = (len(data) - 1) // block
    total_nll, total_tok = 0.0, 0
    for start in tqdm(range(0, n_windows, bs), desc="perplexity"):
        idx = range(start, min(start + bs, n_windows))
        x = torch.stack([torch.from_numpy(data[i * block:(i + 1) * block].astype(np.int64)) for i in idx])
        y = torch.stack([torch.from_numpy(data[i * block + 1:(i + 1) * block + 1].astype(np.int64)) for i in idx])
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = logits_of(model, x)
        nll = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_nll += nll.item()
        total_tok += y.numel()
    mean_nll = total_nll / total_tok
    return math.exp(mean_nll), mean_nll


# --------------------------------------------------- MULTIPLE CHOICE (LL)
@torch.no_grad()
def cont_logprob(model, ctx_ids, cont_ids, device, max_len):
    """Sum of log P(continuation | context)."""
    ids = (ctx_ids + cont_ids)[-max_len:]
    x = torch.tensor([ids], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        logits = logits_of(model, x)
    lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    tok_lp = lp.gather(1, x[0, 1:, None]).squeeze(1)
    return tok_lp[-len(cont_ids):].sum().item()


def run_mc(model, examples, device, max_len, name):
    """examples: list of (context, [continuations], gold_index)."""
    correct = correct_norm = 0
    for ctx, conts, gold in tqdm(examples, desc=name):
        ctx_ids = encode(ctx, bos=True)
        scores, scores_norm = [], []
        for c in conts:
            lp = cont_logprob(model, ctx_ids, encode(c), device, max_len)
            scores.append(lp)
            scores_norm.append(lp / len(c))  # per-character: fixes length bias
        correct += int(np.argmax(scores) == gold)
        correct_norm += int(np.argmax(scores_norm) == gold)
    n = len(examples)
    return {"acc": correct / n, "acc_norm": correct_norm / n, "n": n}


def hellaswag(limit):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if limit:
        ds = ds.select(range(limit))
    return [
        (f"{r['activity_label']}: {r['ctx']}", [" " + e.strip() for e in r["endings"]], int(r["label"]))
        for r in ds
    ]


def arc_easy(limit):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    if limit:
        ds = ds.select(range(limit))
    out = []
    for r in ds:
        labels = r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        out.append((f"Question: {r['question']}\nAnswer:",
                    [" " + t for t in r["choices"]["text"]],
                    labels.index(r["answerKey"])))
    return out


# ------------------------------------------------------------------- MAIN
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "checkpoint_latest.pt"),
                   help="PRETRAINED checkpoint, not an SFT one")
    p.add_argument("--val", default=str(ROOT / "data" / "val.bin"), help="held-out val.bin (uint16 memmap)")
    p.add_argument("--block", type=int, default=MAX_CTX)
    p.add_argument("--limit", type=int, default=0, help="cap MC examples (0 = all)")
    p.add_argument("--out", default="eval_results.json")
    args = p.parse_args()
    assert args.block <= MAX_CTX, f"--block must be <= {MAX_CTX} (RoPE table size)"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    ppl, nll = perplexity(model, args.val, args.block, device)
    results = {"perplexity": ppl, "val_loss": nll}
    results["hellaswag"] = run_mc(model, hellaswag(args.limit), device, args.block, "hellaswag")
    results["arc_easy"] = run_mc(model, arc_easy(args.limit), device, args.block, "arc_easy")

    print(json.dumps(results, indent=2))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()