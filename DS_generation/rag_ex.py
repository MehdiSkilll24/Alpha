# make_rag_data.py
import json, os
from pathlib import Path
from datasets import load_dataset
from rag import retrieve          # loads your index.pt

D = Path("/content/drive/MyDrive/Transformers/Alpha/DS_generation") if os.path.exists("/content") else Path(__file__).parent.absolute()

tq = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
tq = tq.shuffle(seed=42).select(range(20000, 26000))   # different slice from search_ex

rag_ex = []
for ex in tq:
    q = ex["question"]
    ctx = "\n\n".join(c for c, _ in retrieve(q, 3))
    names = [ex["answer"]["value"]] + ex["answer"]["aliases"]
    names = [n.lower() for n in names if len(n) >= 3]      # skip tiny aliases
    if any(n in ctx.lower() for n in names):                # answer must be in the passages
        rag_ex.append({
            "instruction": "Answer the question using the passages below.",
            "input": f"{ctx}\n\nQuestion: {q}",
            "output": ex["answer"]["value"],
        })

with open(D / "rag_ex.json", "w", encoding="utf-8") as f:
    json.dump(rag_ex, f)

print(len(rag_ex), "kept of", len(tq))