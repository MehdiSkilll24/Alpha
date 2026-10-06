# make_search_data.py
import json, os
from pathlib import Path
from datasets import load_dataset

D = Path("/content") if os.path.exists("/content") else Path(__file__).parent.absolute()

tq = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
tq = tq.shuffle(seed=42).select(range(15000))

search_ex = [
    {"instruction": ex["question"], "input": "",
     "output": f"[SEARCH] {ex['question']}"}
    for ex in tq
]

with open(D / "search_ex.json", "w", encoding="utf-8") as f:
    json.dump(search_ex, f)

print(len(search_ex), "->", D / "search_ex.json")