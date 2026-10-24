# rag.py
import time
import torch
from ddgs import DDGS
from prompts import INSTR
from pathlib import Path
import re
from wordsegment import load, segment, UNIGRAMS

dev = "cuda" if torch.cuda.is_available() else "cpu"

def fix_glue(text):
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
    ctx = web_res

    return (f"### Instruction:\n{INSTR}\n\n"                         # must match sft.py's fmt() exactly: "### Instruction:\n{instr}\n\n### Input:\n..."
            f"### Input:\n{ctx}\n\nQuestion: {q}\n\n### Response:\n"), ctx


def chunk(text, size=150, overlap=30):
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, max(1, len(w) - overlap), size - overlap)]


def junky(c):                                   # mostly table syntax -> skip
    w = c.split()
    return not w or c.count("|") / len(w) > 0.15


def web_retrieve(q, n_pages=3, max_retries=3):
    # 1. search, with retries
    res = None
    for attempt in range(max_retries):          # set max_retries = 2
        try:
            res = list(DDGS(timeout=8).text(q, max_results=n_pages, backend="bing,brave,yahoo") or [])
            if res:
                break
        except Exception as e:
            print(f"[SEARCH ERROR {attempt + 1}/{max_retries}] {type(e).__name__}: {e}")
            time.sleep(1 if "No results" in str(e) else 2 ** attempt)   # no-results is fast, retry quickly

    if res is None:
        print("[RETRIEVAL FAILED] Search failed after all retries.")
        return []
    if not res: # <=> not None (res = empty list != None)
        print("[NO RESULTS] Search returned zero results.")
        return []

    return fix_glue(res[0].get("body", ""))
