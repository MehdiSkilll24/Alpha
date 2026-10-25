"""Web search for the assistant: first-result snippet via ddgs, with glued-word repair."""
import re
import time

from ddgs import DDGS
from wordsegment import UNIGRAMS, load, segment

from prompts import INSTR

if not UNIGRAMS:        # wordsegment's data must be loaded once, or segment() raises ValueError
    load()


def fix_glue(text):
    """Split words that search snippets sometimes glue together ("whichcame" -> "which came")."""
    def repl(m):
        s = m.group(0)
        if s.lower() in UNIGRAMS:                  # real word, leave it
            return s
        out, i = [], 0
        for w in segment(s.lower()):               # split, then restore original casing
            out.append(s[i:i + len(w)])
            i += len(w)
        return " ".join(out)
    return re.sub(r"[A-Za-z]{6,}", repl, text)


def build_prompt(q, web_res):
    """Reader prompt. The format must match sft.py's fmt() exactly."""
    return (f"### Instruction:\n{INSTR}\n\n"
            f"### Input:\n{web_res}\n\nQuestion: {q}\n\n### Response:\n"), web_res


def web_retrieve(q, n_pages=1, max_retries=3):
    """Return the first search result's snippet (glue-repaired), or "" if nothing was found.

    n_pages is how many results are requested; only the first one's snippet is used.
    """
    res = None
    for attempt in range(max_retries):
        try:
            res = list(DDGS(timeout=8).text(q, max_results=n_pages, backend="bing,brave,yahoo") or [])
            if res:
                break
            time.sleep(1)                           # empty result: retry quickly
        except Exception as e:
            print(f"[SEARCH ERROR {attempt + 1}/{max_retries}] {type(e).__name__}: {e}")
            time.sleep(1 if "No results" in str(e) else 2 ** attempt)

    if res is None:
        print("[RETRIEVAL FAILED] Search failed after all retries.")
        return ""
    if not res:
        print("[NO RESULTS] Search returned zero results.")
        return ""

    return fix_glue(res[0].get("body", ""))