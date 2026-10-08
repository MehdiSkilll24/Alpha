# retrieve.py
import torch
from sentence_transformers import SentenceTransformer
import trafilatura
from ddgs import DDGS

dev = "cuda" if torch.cuda.is_available() else "cpu"
emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)

def build_prompt(q, hits):
    ctx = "\n\n".join(p for p, _ in hits)
    return (f"### Instruction:\nAnswer the question using the passages below.\n\n"
            f"### Input:\n{ctx}\n\nQuestion: {q}\n\n### Response:\n"), ctx

def chunk(text, size=300, overlap=30):
    w = text.split()
    return [" ".join(w[i:i+size]) for i in range(0, max(1, len(w) - overlap), size - overlap)]

def web_retrieve(q, k=3, n_pages=5):
    try:
        res = DDGS().text(q, max_results=n_pages)
    except Exception:
        return []
    pool = []
    for r in res:
        html = trafilatura.fetch_url(r["href"])
        txt = (trafilatura.extract(html) if html else None) or r["body"]   # snippet if fetch fails
        pool += [f"{r['title']}. {c}" for c in chunk(txt)[:20]]
    if not pool:
        return []
    Ew = emb.encode(pool, batch_size=64, normalize_embeddings=True, convert_to_tensor=True)
    qv = emb.encode("Represent this sentence for searching relevant passages: " + q,
                    normalize_embeddings=True, convert_to_tensor=True)
    top = torch.topk(Ew @ qv, min(k, len(pool)))
    return [(pool[i], s.item()) for s, i in zip(top.values, top.indices)]


