"""TriviaQA end-to-end eval for Alpha: closed-book vs always-retrieve vs routed.

Systems (all graded on the same questions, the ones with cached web hits):
  closed_book     base model, 5-shot "Question/Answer" prompt, no retrieval
  always_retrieve SFT model, retrieval for EVERY question, reader prompt from search.build_prompt
  routed          the full production path, inference.answer() (router -> search -> reader -> grounding check)

Web results come from the cache built by prefetch_hits.py, so every run is reproducible and
needs no network. Metrics: contains (headline), EM, token F1, all against TriviaQA aliases.

Progress is written to a .partial.jsonl file after every question, so a Colab disconnect only
costs the question in flight; rerunning the same command resumes (use --fresh to start over).

Run from model/ (next to inference.py):
  python eval_trivia_rag.py --sft ../checkpoints/checkpoint_sftv5.pt \
      --base ../checkpoints/checkpoint_latest.pt --cache hits_snippets.json --limit 30   # smoke test
  python eval_trivia_rag.py --sft ../checkpoints/checkpoint_sftv5.pt \
      --base ../checkpoints/checkpoint_latest.pt --cache hits_snippets.json              # full run
"""
import argparse
import contextlib
import io
import json
import math
import random
import re
import string
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.absolute().parent

REFUSALS = {"I couldn't find that.", "Search failed, no results.", "Couldn't parse that expression."}
FEWSHOT = ("Answer each question with a short answer.\n\n" + "".join(
    f"Question: {q}\nAnswer: {a}\n\n" for q, a in [
        ("What is the capital of France?", "Paris"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
        ("What is the chemical symbol for gold?", "Au"),
        ("In which year did World War II end?", "1945"),
        ("Which planet is known as the Red Planet?", "Mars")]))


# ------------------------------------------------------------------ data
def load_questions(n=1000, seed=0):
    """Same sample as your earlier plan: TriviaQA validation, shuffle seed 0, first 1000."""
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").shuffle(seed=seed).select(range(n))
    out = []
    for ex in ds:
        a = ex["answer"]
        out.append({"q": ex["question"].strip(), "answers": [a["value"]] + list(a["aliases"])})
    return out


# --------------------------------------------------------------- metrics
_PUNCT = set(string.punctuation)


def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in _PUNCT)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def contains(pred, answers):
    """Any alias appears in the prediction as whole tokens (no 'art' in 'party')."""
    p = f" {norm(pred)} "
    return any(len(n) >= 2 and f" {n} " in p for n in map(norm, answers))


def exact_match(pred, answers):
    p = norm(pred)
    return any(p == norm(a) for a in answers)


def f1(pred, answers):
    pt, best = norm(pred).split(), 0.0
    for a in answers:
        at = norm(a).split()
        same = sum((Counter(pt) & Counter(at)).values())
        if same:
            pr, rc = same / len(pt), same / len(at)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0


# -------------------------------------------------------------- plumbing
@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class HitCache:
    """Drop-in replacement for search.web_retrieve that reads the prefetched cache."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.d = json.load(f)
        self.called = False

    def __call__(self, q):
        self.called = True
        return self.d.get(q)


def load_partial(path, fresh):
    done = {}
    if path.exists() and fresh:
        path.unlink()
    elif path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # last line may be cut off by a disconnect
                done[r["q"]] = r
        print(f"Resuming: {len(done)} questions already done ({path.name})")
    return done


def main():
    import torch
    from tqdm import tqdm

    import model.inference as inf  # the production code path

    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default= ROOT/ "checkpoints" / "checkpoint_sftv5.pt", help="SFT checkpoint (router + reader)")
    ap.add_argument("--base", default= ROOT/ "checkpoints" / "checkpoint_latest.pt", help="pretrained checkpoint for the closed-book baseline")
    ap.add_argument("--cache", default= ROOT / "jsons" / "hits_snippets.json")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N questions")
    ap.add_argument("--fresh", action="store_true", help="ignore a previous partial run")
    ap.add_argument("--out", default="trivia_rag_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = HitCache(args.cache)
    usable = sum(1 for v in cache.d.values() if v)
    print(f"Cache: {Path(args.cache).absolute()}\n  {len(cache.d)} entries, {usable} with hits")
    items = [e for e in load_questions(args.n, args.seed) if cache.d.get(e["q"])]
    if args.limit:
        items = items[:args.limit]
    print(f"Grading on {len(items)} questions that have cached web hits")
    if not args.limit and len(items) < 0.85 * args.n:
        print(f"  WARNING: expected ~{args.n} minus failed searches. Is this the finished cache? "
              "Small n means wide confidence intervals.")

    tag = f"{Path(args.sft).stem}__{Path(args.base).stem}__{Path(args.cache).stem}{len(cache.d)}"
    partial = Path(args.out).with_name(f"{Path(args.out).stem}.{tag}.partial.jsonl")
    done = load_partial(partial, args.fresh)

    base_model, base_tok = inf.load_model(args.base, device)
    sft_model, sft_tok = inf.load_model(args.sft, device)
    # inference.py keeps these as module globals (set in its __main__); mirror its defaults
    inf.model, inf.tok, inf.device = sft_model, sft_tok, device
    inf.a = argparse.Namespace(max_new=200, temp=0.3, top_k=20, top_p=0.95, rep=1.0)
    inf.web_retrieve = cache  # answer() looks this name up at call time

    rows = []
    with open(partial, "a", encoding="utf-8") as pf:
        for ex in tqdm(items, desc="trivia"):
            q, answers = ex["q"], ex["answers"]
            if q in done:
                rows.append(done[q])
                continue
            hits = cache.d[q]

            prompt = f"{FEWSHOT}Question: {q}\nAnswer:"
            cb = inf.generate(base_model, base_tok, prompt, device, 16, 0, 0, 1.0, 1.0)
            cb = cb.split("\n")[0].strip()

            reader_prompt, ctx = inf.build_prompt(q, hits)
            ar = inf.gen(reader_prompt, temp=0, rep=1.0)

            cache.called = False
            with quiet():
                routed = inf.answer(q, [])

            row = {"q": q, "answers": answers, "closed_book": cb, "always_retrieve": ar,
                   "routed": routed, "searched": cache.called, "ctx_has_answer": contains(ctx, answers)}
            rows.append(row)
            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
            pf.flush()

    report(rows)
    show_examples(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f"\nAll answers saved to {args.out}")


def score(rows, key):
    n = len(rows)
    return (sum(contains(r[key], r["answers"]) for r in rows) / n,
            sum(exact_match(r[key], r["answers"]) for r in rows) / n,
            sum(f1(r[key], r["answers"]) for r in rows) / n)


def report(rows):
    n = len(rows)
    print(f"\nTriviaQA validation, n={n}\n")
    print(f"{'system':18} {'contains':>16} {'EM':>7} {'F1':>7}")
    for key in ("closed_book", "always_retrieve", "routed"):
        c, e, f = score(rows, key)
        print(f"{key:18} {c:9.1%} ±{ci95(c, n):4.1%} {e:7.1%} {f:7.1%}")

    has = [r for r in rows if r["ctx_has_answer"]]
    no = [r for r in rows if not r["ctx_has_answer"]]
    ar_has = sum(contains(r["always_retrieve"], r["answers"]) for r in has) / max(len(has), 1)
    ar_no = sum(contains(r["always_retrieve"], r["answers"]) for r in no) / max(len(no), 1)
    answered = [r for r in rows if r["routed"] not in REFUSALS]
    routed_ok = sum(contains(r["routed"], r["answers"]) for r in answered) / max(len(answered), 1)
    print("\nWhere it fails (always_retrieve):")
    print(f"  retrieval recall (answer inside the retrieved text): {len(has) / n:.1%}")
    print(f"  reader accuracy when the answer IS in the text:      {ar_has:.1%}  (n={len(has)})")
    print(f"  accuracy when it is NOT in the text:                 {ar_no:.1%}  (n={len(no)})")
    print("\nRouted system:")
    print(f"  chose to search:          {sum(r['searched'] for r in rows) / n:.1%}")
    print(f"  refused / said not found: {1 - len(answered) / n:.1%}")
    print(f"  accuracy when it answered: {routed_ok:.1%}  (n={len(answered)})")


def show_examples(rows, k=6):
    """A few random questions with what each system said, to see the failure modes by eye."""
    print("\nSample answers (gold alias first):")
    for r in random.Random(0).sample(rows, min(k, len(rows))):
        mark = lambda key: "✓" if contains(r[key], r["answers"]) else "✗"
        print(f"\n  Q: {r['q'][:110]}\n  gold: {r['answers'][0]} | text had answer: {r['ctx_has_answer']}")
        for key in ("closed_book", "always_retrieve", "routed"):
            print(f"  {mark(key)} {key:15} {r[key][:110]!r}")


if __name__ == "__main__":
    main()