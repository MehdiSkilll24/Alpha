# rag.py
import time
import torch
from sentence_transformers import SentenceTransformer
import trafilatura
from ddgs import DDGS
from prompts import INSTR

dev = "cuda" if torch.cuda.is_available() else "cpu"
emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)


def build_prompt(q, hits):
    ctx = "\n\n".join(p for p, _ in hits)
    # must match sft.py's fmt() exactly: "### Instruction:\n{instr}\n\n### Input:\n..."
    return (f"### Instruction:\n{INSTR}\n\n"
            f"### Input:\n{ctx}\n\nQuestion: {q}\n\n### Response:\n"), ctx


def chunk(text, size=150, overlap=30):
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, max(1, len(w) - overlap), size - overlap)]


def junky(c):                                   # mostly table syntax -> skip
    w = c.split()
    return not w or c.count("|") / len(w) > 0.15


def web_retrieve(q, k=3, n_pages=5, max_retries=3):
    # 1. search, with retries
    res = None
    for attempt in range(max_retries):
        try:
            res = list(DDGS().text(q, max_results=n_pages) or [])
            break
        except Exception as e:
            print(f"[SEARCH ERROR {attempt + 1}/{max_retries}] {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    if res is None:
        print("[RETRIEVAL FAILED] Search failed after all retries.")
        return []
    if not res:
        print("[NO RESULTS] Search returned zero results.")
        return []
    print(f"[SEARCH] {len(res)} results")

    # 2. fetch pages, split into chunks; keep (title, chunk) pairs
    pool = []
    for r in res:
        title = r.get("title") or ""
        url = r.get("href")
        txt = None
        if url:
            try:
                html = trafilatura.fetch_url(url)
                if html:
                    txt = trafilatura.extract(html)
            except Exception as e:
                print(f"[PAGE ERROR] {url}: {type(e).__name__}")
        txt = txt or (r.get("body") or "")          # fall back to the search snippet
        if not txt.strip():
            print(f"[EMPTY PAGE] {url or title}")
            continue
        pool += [(title, c) for c in chunk(txt)[:20] if not junky(c)]

    if not pool:
        print("[NO PASSAGES] No usable text from search results.")
        return []

    # 3. rank with the title included (helps matching), return the chunk text only
    try:
        Ew = emb.encode([f"{t}. {c}" for t, c in pool], batch_size=64,
                        normalize_embeddings=True, convert_to_tensor=True)
        qv = emb.encode("Represent this sentence for searching relevant passages: " + q,
                        normalize_embeddings=True, convert_to_tensor=True)
        top = torch.topk(Ew @ qv, min(k, len(pool)))
        hits = [(pool[int(i)][1], s.item()) for s, i in zip(top.values, top.indices)]
        print(f"[RETRIEVED] {len(hits)} passages from {len(pool)} candidates")
        return hits
    except Exception as e:
        print(f"[EMBEDDING/RANKING ERROR] {type(e).__name__}: {e}")
        return []