# generate.py
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from main import Transformer
from bpe import load_bpe_tokenizer
import json
from search import build_prompt, web_retrieve
import re
from calculator import calc

SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT = SCRIPT_DIR.parent      
TOK = ROOT / "jsons" / "fineweb_bpe.json"
MAX_CTX = 2048  # RoPE table size; the model has never seen positions beyond this
CALL = re.compile(r"\s*\[(SEARCH|CALC)\]\s*(.*)", re.S)

SYSTEM = "You are Alpha, a small assistant with tools built by Mehdi."   # "" to disable

def history_ids(tok, hist, user_msg, system="", reserve=400):
    enc = lambda s: tok.tok.encode(s, add_special_tokens=False).ids
    bos, eos = tok.word_to_idx["<bos>"], tok.word_to_idx["<eos>"]
    def build(turns):
        ids = [bos]
        for i, (u, a) in enumerate(turns):
            head = f"### System:\n{system}\n\n" if (i == 0 and system) else ""
            ids += enc(f"{head}### Instruction:\n{u}\n\n### Response:\n")
            if a is not None:
                ids += enc(a) + [eos]
        return ids
    turns = list(hist) + [(user_msg, None)]
    while len(turns) > 1 and len(build(turns)) > MAX_CTX - reserve:
        turns = turns[1:]                    # drop the oldest turn, never a partial one
    return build(turns)

def grounded(ans, ctx, thr=0.6):
    words = {w for w in re.findall(r"[a-z0-9]{3,}", ans.lower())}
    nums = re.findall(r"\d+(?:\.\d+)?", ans)
    if any(n not in ctx for n in nums): return False
    if not words:
        return False
    c = ctx.lower()
    return sum(w in c for w in words) / len(words) >= thr

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
             top_k=50, top_p=0.95, rep_penalty=1.1, return_flag=False):
    ids = (tok.encode(prompt)[:-1] if isinstance(prompt, str) else list(prompt))[-(MAX_CTX - 1):]
    n_prompt = len(ids)
    budget = min(max_new, MAX_CTX - n_prompt)
    ban = [tok.word_to_idx[t] for t in ("<pad>", "<unk>", "<bos>") if t in tok.word_to_idx]
    eos = tok.word_to_idx.get("<eos>")

    x = torch.tensor([ids], dtype=torch.long, device=device)  # full history (rep penalty + decoding)
    inp, caches = x, None                                     # step 0 feeds the whole prompt (prefill)
    done = False
    for _ in range(budget):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, caches = model(inp, caches=caches,
                                   start_pos=x.size(1) - inp.size(1), return_cache=True)
        logits = logits[:, -1, :].float()

        logits[:, ban] = float("-inf")

        if rep_penalty != 1.0:
            seen = torch.unique(x[0, -256:])
            l = logits[0, seen]
            logits[0, seen] = torch.where(l > 0, l / rep_penalty, l * rep_penalty)

        if temperature == 0:
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
                sp[(cum - sp) > top_p] = 0.0
                probs = torch.zeros_like(probs).scatter_(1, si, sp)
                probs /= probs.sum(-1, keepdim=True)
            nxt = torch.multinomial(probs, 1)

        x = torch.cat([x, nxt], dim=1)
        inp = nxt                                 # from now on, feed ONLY the new token
        if eos is not None and nxt.item() == eos:
            done = True
            break

    text = tok.decode(x[0, n_prompt:].tolist())
    return (text, done) if return_flag else text

def gen(p, temp=None, rep=None, n=None, flag = False):
        
        out = generate(model, tok, p, device,         # out is either a tuple of (text, done) or just = text depending if gen's flag true or false (if called as an answer, then it's true)
                       n or a.max_new,a.temp if temp is None else temp,
                       a.top_k, a.top_p,a.rep if rep is None else rep,
                       return_flag=flag)
        
        if flag:
            return out[0].strip(), out[1]
        return out.strip()
    
def answer(q, hist):
    pids = history_ids(tok, hist, q, SYSTEM)
    p0 = f"### Instruction:\n{q}\n\n### Response:\n"
    first = gen(p0, temp=0, rep=1.0, n=48)        
    m = CALL.match(first)
    if not m: # -> means it's a normal model answer
        for _ in range(3):
            out, done = gen(pids, temp=0.7, rep=1.15, n=400, flag=True) 
            if not CALL.match(out):
                break
        else:
            return first

        if not done and "." in out:
            out = out[:out.rfind(".") + 1]

        return out
    
    cmd, arg = m.group(1), m.group(2).split("\n")[0].strip()
    if cmd == "SEARCH":
        print("Search mode...\n")
        web_res = web_retrieve(q)
        print("raw: ", web_res)                                               # user's text, not the model's copy
        if not web_res:
            return "Search failed, no results."
        prompt, ctx = build_prompt(q, web_res)
        model_res = gen(prompt, temp=0, rep=1.0)                                    # reader: greedy copy
        return model_res if grounded(model_res, ctx) else "I couldn't find that."

    if cmd == "CALC" and arg:
        r = calc(arg)
        return r if r is not None else "Couldn't parse that expression."
    return first

if __name__ == "__main__":
    hist = []
    FAIL = {"Search failed, no results.", "I couldn't find that.", "Couldn't parse that expression."}
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "checkpoint_sftv4.pt"))
    p.add_argument("--prompt", default=None)
    p.add_argument("--max_new", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.3)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--rep", type=float, default=1.0)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.ckpt, device)
    
    if a.prompt:
        print("model: ", answer(a.prompt, []))

    else:
        while True:
            s = input("prompt> ").strip()
            if s in ("quit", "exit"): break
            if s == "": continue
            if s == "/reset": hist.clear(); print("(history cleared)"); continue
            reply = answer(s, hist)
            print("\n" + reply + "\n")
            if reply and reply not in FAIL:
                hist.append((s, reply))