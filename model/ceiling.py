# ceiling.py: how often does any passage contain a gold answer?
import json
from datasets import load_dataset
from metrics import contains
cache = json.load(open("/content/drive/MyDrive/Transformers/Alpha/hits_cache.json"))
ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").shuffle(seed=0).select(range(1000))
rows = [(ex, cache[ex["question"].strip()]) for ex in ds if ex["question"].strip() in cache]
hit = [contains(" ".join(p for p, _ in h), [ex["answer"]["value"]] + ex["answer"]["aliases"]) for ex, h in rows]
print(f"gold in passages: {sum(hit)/len(hit):.3f} of {len(hit)}")