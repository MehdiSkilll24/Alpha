# generate.py
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from main import Transformer
from bpe import load_bpe_tokenizer
import json
from rag import build_prompt, web_retrieve
import re
from calculator import calc
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT = SCRIPT_DIR.parent      
TOK = ROOT / "jsons" / "fineweb_bpe.json"
MAX_CTX = 2048  # RoPE table size; the model has never seen positions beyond this
CALL = re.compile(r"\s*\[(SEARCH|CALC)\]\s*(.*)", re.S)

def load_model(ckpt_path, device):
    tok = load_bpe_tokenizer(TOK)
    with open(ROOT / "jsons" / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt["arch"]  # trust the checkpoint over config.json
    assert arch["vocab_size"] == len(tok), (
        f"Tokenizer has {len(tok)} tokens but checkpoint expects {arch['vocab_size']}. Wrong BPE file?"
    )

    model = Transformer(
        arch["d_model"], arch["num_heads"], arch["vocab_size"],
        arch["d_ff"], arch["num_kv_heads"], arch["num_decoder_layers"],
    )
    model.load_state_dict(ckpt["model_state_dict"])  # strict: fails loudly on mismatch
    model.to(device).eval()
    print(f"Loaded step {ckpt.get('step')} | val loss {ckpt.get('val_loss')}")
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt, device, max_new=200, temperature=0.8,
             top_k=50, top_p=0.95, rep_penalty=1.1):
    ids = tok.encode(prompt)[:-1]  # <-- check: must return a list of ints
    n_prompt = len(ids)
    ban = [tok.word_to_idx[t] for t in ("<pad>", "<unk>", "<bos>") if t in tok.word_to_idx]
    eos = tok.word_to_idx.get("<eos>")

    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        ctx = x[:, -MAX_CTX:]  # no KV cache: recompute the window each step
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(ctx)[:, -1, :].float()

        logits[:, ban] = float("-inf")

        if rep_penalty != 1.0:  # penalize tokens already in the recent context
            seen = torch.unique(x[0, -256:])
            l = logits[0, seen]
            logits[0, seen] = torch.where(l > 0, l / rep_penalty, l * rep_penalty)

        if temperature == 0:  # greedy
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits[logits < kth] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sp, si = probs.sort(descending=True)
                cum = sp.cumsum(-1)
                sp[(cum - sp) > top_p] = 0.0  # keep the token that crosses the threshold
                probs = torch.zeros_like(probs).scatter_(1, si, sp)
                probs /= probs.sum(-1, keepdim=True)
            nxt = torch.multinomial(probs, 1)

        x = torch.cat([x, nxt], dim=1)
        if eos is not None and nxt.item() == eos:
            break

    return tok.decode(x[0, n_prompt:].tolist())  # <-- check: must accept a list of ints


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "checkpoint_sftv3.pt"))
    p.add_argument("--prompt", default=None)
    p.add_argument("--max_new", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.3)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--rep", type=float, default=1.0)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.ckpt, device)
    def gen(p):
        return generate(model, tok, p, device, a.max_new, a.temp, a.top_k, a.top_p, a.rep).strip()
    
    def answer(q):
        out = gen(f"### Instruction:\n{q}\n\n### Response:\n")
        print("FIRST:", repr(out))
        m = CALL.match(out)
        if not m:
            return out
        cmd, arg = m.group(1), m.group(2).split("\n")[0].strip()
        
        if cmd == "SEARCH" and arg:
            hits = web_retrieve(arg)
            if not hits:
                return "Search failed, no results."
            prompt, ctx = build_prompt(arg, hits)
            res = gen(prompt)
            print("PASSAGES:", hits)
            return res if res.strip().lower() in ctx.lower() else "I couldn't find that."
            
        if cmd == "CALC" and arg:
            r = calc(arg)
            if r is not None:
                return r
        return out
    if a.prompt:
        print(answer(a.prompt))
    else:
        while True:
            s = input("prompt> ").strip()
            if s in ("", "quit", "exit"):
                break
            print("\n" + answer(s) + "\n")