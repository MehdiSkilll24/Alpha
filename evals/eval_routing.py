"""Routing eval for Alpha's SFT model: does it pick SEARCH / CALC / plain ANSWER correctly?

Reproduces the FIRST pass of inference.answer(): greedy decoding, 48 tokens, bare
"### Instruction / ### Response" prompt (no system prompt, no history). A prompt
routes to a tool iff the output matches CALL; otherwise it counts as "answer".

Usage (from the model/ folder):
  python eval_routing.py --ckpt ../checkpoints/checkpoint_sftv4.pt \
      --prompts routing_prompts.json --out routing_sftv4.json \
      --train-files ../jsons/search_ex.json ../jsons/calc_ex.json ../jsons/chat.json ../jsons/identity.json
"""

import argparse
import json
import re
from collections import defaultdict

import torch
from tqdm import tqdm

from model.inference import CALL, generate, load_model  # same regex + decoding as production
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
LABELS = ["search", "calc", "answer"]
ROOT = SCRIPT_DIR.parent                        # .../Alpha


def route(model, tok, device, q):
    p0 = f"### Instruction:\n{q}\n\n### Response:\n"
    # positional args: max_new=48, temperature=0 (greedy), top_k, top_p, rep_penalty=1.0
    first = generate(model, tok, p0, device, 48, 0, 20, 0.95, 1.0).strip()
    m = CALL.match(first)
    return (m.group(1).lower() if m else "answer"), first


# ------------------------------------------------------------ leakage check
def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def instruction_of(ex):
    if "messages" in ex:
        return next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
    return ex.get("instruction", "")


def load_train_prompts(paths):
    texts = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            texts += [instruction_of(ex) for ex in json.load(f)]
    return [words(t) for t in texts if t]


def is_leaked(prompt, train, thr):
    w = words(prompt)
    return bool(w) and any(tw and len(w & tw) / len(w | tw) >= thr for tw in train)


# ------------------------------------------------------------------ report
def report(rows):
    conf = {g: {p: 0 for p in LABELS} for g in LABELS}
    for r in rows:
        conf[r["label"]][r["pred"]] += 1
    n = len(rows)
    acc = sum(r["label"] == r["pred"] for r in rows) / n
    print(f"\nOverall accuracy: {acc:.1%} (n={n})")

    print("\nConfusion (rows = gold, cols = predicted):")
    print(f"{'':10}" + "".join(f"{p:>9}" for p in LABELS))
    for g in LABELS:
        print(f"{g:10}" + "".join(f"{conf[g][p]:>9}" for p in LABELS))

    print("\nPer class:")
    for c in LABELS:
        tp = conf[c][c]
        gold_n = sum(conf[c].values())
        pred_n = sum(conf[g][c] for g in LABELS)
        rec = tp / gold_n if gold_n else float("nan")
        prec = tp / pred_n if pred_n else float("nan")
        print(f"  {c:7} precision {prec:.1%} | recall {rec:.1%} | n={gold_n}")

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[(r["label"], r.get("category", "-"))].append(r["label"] == r["pred"])
    print("\nBy category:")
    for (lab, cat), v in sorted(by_cat.items()):
        print(f"  {lab:7} {cat:14} {sum(v)}/{len(v)}")

    if rows[0]["leaked"] is not None:
        clean = [r for r in rows if not r["leaked"]]
        n_leak = n - len(clean)
        if clean:
            a = sum(r["label"] == r["pred"] for r in clean) / len(clean)
            print(f"\nNear-duplicates of training prompts: {n_leak}. "
                  f"Accuracy on the {len(clean)} non-leaked prompts: {a:.1%}")

    print("\nMisses:")
    for r in rows:
        if r["label"] != r["pred"]:
            print(f"  [{r['label']} -> {r['pred']}] {r['prompt']!r}\n      model: {r['raw'][:80]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, default=str(ROOT / "checkpoints" / "checkpoint_sftv5.pt"))
    ap.add_argument("--prompts", default="routing_prompts.json")
    ap.add_argument("--train-files", nargs="*", default=[], help="SFT json files for the leakage check")
    ap.add_argument("--thr", type=float, default=0.7, help="word-Jaccard threshold for 'near-duplicate'")
    ap.add_argument("--out", default="routing_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.ckpt, device)

    with open(args.prompts, encoding="utf-8") as f:
        items = json.load(f)
    train = load_train_prompts(args.train_files) if args.train_files else None

    rows = []
    for it in tqdm(items, desc="routing"):
        pred, raw = route(model, tok, device, it["prompt"])
        rows.append({**it, "pred": pred, "raw": raw,
                     "leaked": is_leaked(it["prompt"], train, args.thr) if train else None})

    report(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()