# sft.py
import json, math, random, argparse
from pathlib import Path
import torch, torch.nn as nn
from datasets import load_dataset
from main import Transformer
from bpe import load_bpe_tokenizer
import os 
import re

D = Path("/content/drive/MyDrive/Transformers/Alpha") if os.path.exists("/content") else Path(__file__).parent.absolute()
MAX_LEN, MICRO, ACCUM = 2048, 16, 2
LR, WARMUP, EPOCHS, EVAL_EVERY = 3e-5, 50, 3, 200

def fmt(ex):
    if ex.get("input"):
        return (f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n### Response:\n")
    return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n"

def build(ds, tok):
    bos, eos = tok.word_to_idx["<bos>"], tok.word_to_idx["<eos>"]
    enc = lambda s: tok.tok.encode(s, add_special_tokens=False).ids
    out = []
    for ex in ds:
        p = [bos] + enc(fmt(ex))
        a = enc(ex["output"].strip()) + [eos]
        if len(p) + len(a) > MAX_LEN or len(a) < 2:
            continue  # drop, never truncate: a cut answer would lose its <eos>
        out.append((p + a, [-100] * len(p) + a))
    return out

def collate(batch, pad):
    n = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), n), pad)
    lab = torch.full((len(batch), n), -100)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
    return ids[:, :-1], lab[:, 1:]  # right-padding is safe: causal attention

def lr_at(s, total):
    if s < WARMUP: return LR * (s + 1) / WARMUP
    p = (s - WARMUP) / max(1, total - WARMUP)
    return 0.1 * LR + 0.5 * 0.9 * LR * (1 + math.cos(math.pi * p))

@torch.no_grad()
def evaluate(model, data, pad, dev):
    model.eval(); tot, cnt = 0.0, 0
    for i in range(0, len(data), MICRO):
        x, y = collate(data[i:i + MICRO], pad)
        x, y = x.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lg = model(x)
        tot += nn.functional.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1),
                                           ignore_index=-100, reduction="sum").item()
        cnt += (y != -100).sum().item()
    model.train()
    return tot / cnt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(D / "checkpoints" / "checkpoint_latest.pt"))
    ap.add_argument("--out", default=str(D / "checkpoints" / "checkpoint_sft.pt"))
    a = ap.parse_args()

    random.seed(42); torch.manual_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_bpe_tokenizer(D / "fineweb_bpe.json")
    pad = tok.word_to_idx["<pad>"]

    alpaca = list(load_dataset("yahma/alpaca-cleaned")["train"])
    MATH = re.compile(r"^(calculate|compute|evaluate|solve|add|subtract|multiply|divide|"
                  r"find the (sum|product|difference|quotient|result)|"
                  r"what is the (sum|product|difference|quotient|result)|"
                  r"what('s| is) [\d\.\-\+\*/%\s]+)", re.I)

    before = len(alpaca)
    alpaca = [ex for ex in alpaca if ex["input"] or not MATH.match(ex["instruction"].strip())]
    print(f"removed {before - len(alpaca)} math examples")

    FACT = re.compile(r"^(who|when|where|which|what is|what was|what are|how many|how much|in which|name the)\b", re.I)
    before = len(alpaca)
    alpaca = [ex for ex in alpaca
              if ex["input"] or not FACT.match(ex["instruction"].strip())]
    print(f"removed {before - len(alpaca)} alpaca examples")

    with open(D / "search_ex.json", encoding="utf-8") as f:
        search = json.load(f)

    with open(D / "calc_ex.json", encoding="utf-8") as f:
            calc = json.load(f)

    with open(D / "rag_ex.json", encoding="utf-8") as f:
        rag = json.load(f)
        print(len(rag), "->", len(build(rag, tok)), "survive MAX_LEN")

    raw = alpaca + search + calc + rag
    random.shuffle(raw)
    data = build(raw, tok)
    val, train = data[:1000], data[1000:]
    print(f"train {len(train)} | val {len(val)} examples")

    ck = torch.load(a.base, map_location="cpu"); arch = ck["arch"]
    model = Transformer(arch["d_model"], arch["num_heads"], arch["vocab_size"],
                        arch["d_ff"], arch["num_kv_heads"], arch["num_decoder_layers"])
    model.load_state_dict(ck["model_state_dict"]); model.to(dev).train()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=dev.type == "cuda")
    per_step = MICRO * ACCUM
    total = EPOCHS * (len(train) // per_step)
    print(f"base val (answer tokens): {evaluate(model, val, pad, dev):.4f}")

    step, best = 0, float("inf")
    for ep in range(EPOCHS):
        random.shuffle(train)
        for b in range(0, len(train) - per_step + 1, per_step):
            for g in opt.param_groups: g["lr"] = lr_at(step, total)
            opt.zero_grad(set_to_none=True); tl = 0.0
            for m in range(ACCUM):
                x, y = collate(train[b + m * MICRO: b + (m + 1) * MICRO], pad)
                x, y = x.to(dev), y.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lg = model(x)
                loss = nn.functional.cross_entropy(lg.float().reshape(-1, lg.size(-1)),
                                                   y.reshape(-1), ignore_index=-100)
                (loss / ACCUM).backward(); tl += loss.item() / ACCUM
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
            if step % 20 == 0:
                print(f"ep {ep+1} step {step}/{total} | loss {tl:.4f} | gnorm {gn:.2f}")
            if step % EVAL_EVERY == 0 or step == total:
                v = evaluate(model, val, pad, dev)
                print(f"[Eval] step {step} | val {v:.4f}")
                if v < best:
                    best = v
                    torch.save({"model_state_dict": model.state_dict(), "arch": arch,
                                "step": step, "val_loss": v}, a.out)
                    print("  saved best")