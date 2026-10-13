# retrieve.py
import torch
from sentence_transformers import SentenceTransformer
import trafilatura
from ddgs import DDGS
from prompts import INSTR
import time

dev = "cuda" if torch.cuda.is_available() else "cpu"
emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)

def build_prompt(q, hits):
    ctx = "\n\n".join(p for p, _ in hits)
    return (f"### Instruction:{INSTR}\n\n"
            f"### Input:\n{ctx}\n\nQuestion: {q}\n\n### Response:\n"), ctx

def chunk(text, size=150 , overlap=30):
    w = text.split()
    return [" ".join(w[i:i+size]) for i in range(0, max(1, len(w) - overlap), size - overlap)]


def web_retrieve(q, k=3, n_pages=5, max_retries=3):
    def junky(c):
        w = c.split()
        return not w or c.count("|") / len(w) > 0.15
    # 1. Search with retries
    res = None

    for attempt in range(max_retries):
        try:
            res = list(DDGS().text(q, max_results=n_pages) or [])
            break
        except Exception as e:
            print(
                f"[SEARCH ERROR {attempt + 1}/{max_retries}] "
                f"{type(e).__name__}: {e}"
            )

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    if res is None:
        print("[RETRIEVAL FAILED] Search failed after all retries.")
        return []

    if not res:
        print("[NO RESULTS] Search returned zero results.")
        return []

    print(f"[SEARCH] {len(res)} results")

    # 2. Fetch pages and extract text
    pool = []

    for r in res:
        title = r.get("title") or ""
        url = r.get("href")
        snippet = r.get("body") or ""

        txt = None

        if url:
            try:
                html = trafilatura.fetch_url(url)
                if html:
                    txt = trafilatura.extract(html)
            except Exception as e:
                print(f"[PAGE ERROR] {url}: {type(e).__name__}")

        # Fall back to search snippet if page extraction fails
        txt = txt or snippet

        if not txt.strip():
            print(f"[EMPTY PAGE] {url or title}")
            continue

        try:
            chunks = chunk(txt)
        except Exception as e:
            print(f"[CHUNK ERROR] {title}: {type(e).__name__}")
            continue
        
        
        pool.append((r["title"], c for c in chunk(txt)[:20] if not junky(c)))                                    # keep both
        Ew = emb.encode([f"{t}. {c}" for t, c in pool], ...)            # the title helps ranking
        return [(pool[i][1], s.item()) for s, i in zip(top.values, top.indices)]   # reader sees no title

    if not pool:
        print("[NO PASSAGES] No usable text from search results.")
        return []

    # 3. Embed passages and query
    try:
        Ew = emb.encode(
            pool,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_tensor=True
        )

        qv = emb.encode(
            "Represent this sentence for searching relevant passages: " + q,
            normalize_embeddings=True,
            convert_to_tensor=True
        )

        # 4. Rank passages by cosine similarity
        scores = Ew @ qv
        n = min(k, len(pool))
        top = torch.topk(scores, n)

        hits = [
            (pool[i], score.item())
            for score, i in zip(top.values, top.indices)
        ]

        print(f"[RETRIEVED] {len(hits)} passages from {len(pool)} candidates")
        return hits

    except Exception as e:
        print(f"[EMBEDDING/RANKING ERROR] {type(e).__name__}: {e}")
        return []




