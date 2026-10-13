# eval_qa.py
# Compares three systems on TriviaQA validation, using ONLY pre-fetched web hits
# (hits_cache.json from prefetch.py), so every system sees identical passages.
import json, re, random, argparse
from pathlib import Path
from datasets import load_dataset
import torch
from inference import load_model, generate
from rag import build_prompt
from metrics import em, f1, contains

CALL = re.compile(r"\s*\[(SEARCH|CALC)\]\s*(.*)", re.S)
CACHE = Path("/content/drive/MyDrive/Transformers/Alpha/hits_cache.json")
cache = json.load(open(CACHE)) if CACHE.exists() else {}
READ_TOKENS = 150                          # max answer length for the reader


def hits_for(query):
    return cache.get(query.strip(), [])    # cache only: no live web calls


def gen(model, tok, dev, prompt, n):       # greedy decoding: no sampling noise
    return generate(model, tok, prompt, dev, n, 0, 0, 1.0, 1.0).strip()


def read(model, tok, dev, query):
    hits = hits_for(query)
    if not hits:
        return ""                          # cache miss counts as wrong
    prompt, _ = build_prompt(query, hits)
    return gen(model, tok, dev, prompt, READ_TOKENS)


def sys_routed(model, tok, dev, q):
    out = gen(model, tok, dev, f"### Instruction:\n{q}\n\n### Response:\n", 64)
    m = CALL.match(out)
    if not m:
        return "DIRECT", out
    cmd, arg = m.group(1), m.group(2).split("\n")[0].strip()
    if cmd == "SEARCH" and arg:
        return "SEARCH", read(model, tok, dev, arg)   # empty if the model rewrote the query
    return cmd, out                        # [CALC] on a trivia question = routing error


def sys_always(model, tok, dev, q):
    return "SEARCH", read(model, tok, dev, q)


FEW = ("Q: What is the capital of France?\nA: Paris\n\n"
       "Q: Who wrote Hamlet?\nA: William Shakespeare\n\n"
       "Q: What is the chemical symbol for gold?\nA: Au\n\n")


def sys_closed(base, tok, dev, q):
    out = gen(base, tok, dev, FEW + f"Q: {q}\nA:", 16)
    return "DIRECT", out.split("\n")[0]


def ci(flags, B=2000, seed=0):             # 95% bootstrap interval for a proportion
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(rng.choices(flags, k=n)) / n for _ in range(B))
    return means[int(.025 * B)], means[int(.975 * B)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--base", required=True)           # checkpoint_latest.pt
    ap.add_argument("--n", type=int, default=1000)     # must match prefetch.py
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft, tok = load_model(a.sft, dev)
    base, _ = load_model(a.base, dev)

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    ds = ds.shuffle(seed=0).select(range(a.n))
    ds = [ex for ex in ds if ex["question"].strip() in cache]   # grade only cached questions
    print(f"grading on {len(ds)} of {a.n} questions")

    systems = {"closed-book": lambda q: sys_closed(base, tok, dev, q),
               "always-retrieve": lambda q: sys_always(sft, tok, dev, q),
               "routed": lambda q: sys_routed(sft, tok, dev, q)}
    stats = {k: {"hit": [], "em": 0, "f1": 0.0, "search": 0} for k in systems}
    empty = {}

    out_path = CACHE.parent / f"results_{Path(a.sft).stem}.jsonl"   # one file per checkpoint
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            q = ex["question"]
            golds = [ex["answer"]["value"]] + ex["answer"]["aliases"]
            for name, fn in systems.items():
                route, pred = fn(q)
                if name != "closed-book" and pred == "":
                    empty[name] = empty.get(name, 0) + 1
                s = stats[name]
                s["hit"].append(int(contains(pred, golds)))
                s["em"] += em(pred, golds)
                s["f1"] += f1(pred, golds)
                s["search"] += route == "SEARCH"
                f.write(json.dumps({"q": q, "sys": name, "route": route, "pred": pred,
                                    "gold": ex["answer"]["value"], "golds": golds}) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(i + 1, "done")

    n = len(ds)
    print(f"\nresults -> {out_path}")
    print(f"empty (always-retrieve should be 0; routed = rewritten-query cache misses): {empty}")
    print(f"\n{'system':<16}{'contains':>10}{'95% CI':>18}{'EM':>8}{'F1':>8}{'->search':>10}")
    for k, s in stats.items():
        lo, hi = ci(s["hit"])
        c = sum(s["hit"]) / n
        print(f"{k:<16}{c:>10.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}"
              f"{s['em']/n:>8.3f}{s['f1']/n:>8.3f}{s['search']/n:>10.2f}")