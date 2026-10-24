"""Prefetch web hits for the TriviaQA eval, once, with the CURRENT snippet-only search.

Your old hits_cache.json came from the page-fetching pipeline, so it no longer matches what
production does. This writes a new cache (default hits_snippets.json) keyed by question.
It resumes where it stopped and retries failed questions, so you can just rerun it.

Run from model/ (needs network, next to search.py):
  python prefetch_hits.py --cache hits_snippets.json
"""
import argparse
import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT = SCRIPT_DIR.parent      
JSON = ROOT / "jsons" 

def save(cache, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(path)  # atomic: a Colab disconnect can't leave a half-written file


def main():
    from tqdm import tqdm

    from evals.eval_trivia_rag import load_questions
    from search import web_retrieve

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default= JSON / "hits_snippets.json")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.7, help="seconds between searches (rate limits)")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    path = Path(args.cache)
    cache = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    qs = [e["q"] for e in load_questions(args.n, args.seed)]
    todo = [q for q in qs if not cache.get(q)]
    print(f"{len(qs) - len(todo)} cached, {len(todo)} to fetch")

    for i, q in enumerate(tqdm(todo, desc="prefetch")):
        res = None
        for attempt in range(args.retries):
            try:
                res = web_retrieve(q)
                if res:
                    break
            except Exception as e:
                print(f"\n  retry {attempt + 1} for {q[:50]!r}: {type(e).__name__}")
            time.sleep(2 ** attempt)  # back off before the next attempt
        if res:
            try:
                json.dumps(res)
            except TypeError:
                raise SystemExit(f"web_retrieve returned a {type(res).__name__}, which isn't JSON-serializable. "
                                 "Tell me its type and I'll switch the cache to pickle.")
        cache[q] = res or None
        time.sleep(args.sleep)
        if (i + 1) % 25 == 0:
            save(cache, path)

    save(cache, path)
    ok = sum(1 for q in qs if cache.get(q))
    print(f"\n{ok}/{len(qs)} questions have hits. Rerun to retry the {len(qs) - ok} that failed.")


if __name__ == "__main__":
    main()