# retrieve.py
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

dev = "cuda" if torch.cuda.is_available() else "cpu"
emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)

if os.path.exists("/content"):          # Colab
    INDEX = Path("/content/index.pt")
else:
    INDEX = Path(__file__).parent / "index.pt"


def chunk(text, size=150, overlap=30):
    w = text.split()
    return [" ".join(w[i:i+size] for i in range (0, max(1, len(w) - overlap), size - overlap))]

def build_idx():
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    chunks = []
    for i, a in enumerate(ds):
        if i == 100_000: break
        chunks += chunk(a["text"])[:4]          # first 4 chunks per article keeps it small

    E = emb.encode(chunks, batch_size=256, normalize_embeddings=True,
                   convert_to_tensor=True, show_progress_bar=True)
    torch.save({"chunks": chunks, "E": E.cpu()}, "index.pt")

if not INDEX.exists():
    build_idx()

_idx = torch.load(INDEX)
chunks = _idx["chunks"]

E = _idx["E"].to(dev)

# query
def retrieve(q, k=3):
    qv = emb.encode("Represent this sentence for searching relevant passages: " + q,
                    normalize_embeddings=True, convert_to_tensor=True)
    scores = E @ qv.to(E.device)
    top = torch.topk(scores, k)
    return [(chunks[i], s.item()) for s, i in zip(top.values, top.indices)]

def rag_prompt(q, k=3):
    ctx = "\n\n".join(p for p, _ in retrieve(q, k))
    return (f"### Instruction:\nAnswer the question using the passages below.\n\n"
            f"### Input:\n{ctx}\n\nQuestion: {q}\n\n### Response:\n")

