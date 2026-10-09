# make_rag_long.py (ASQA version)
import json, random
from datasets import load_dataset
from prompts import INSTR
from pathlib import Path
import os
random.seed(0)
D = Path("/content/drive/MyDrive/Transformers/Alpha") if os.path.exists("/content") else Path(__file__).parent.absolute()

ds = load_dataset("din0s/asqa", split="train")

pool = [k["content"] for ex in ds for a in ex["annotations"]
        for k in a["knowledge"] if k["content"]]

rag_asqa = []
for ex in ds:
    ann = ex["annotations"][0]
    ps = [k["content"] for k in ann["knowledge"] if k["content"]][:3]
    if not ps or not ann["long_answer"]:           # skip rows with nothing usable
        continue
    ps += random.sample(pool, 1)
    random.shuffle(ps)
    rag_asqa.append({"instruction": INSTR,
                 "input": "\n\n".join(ps) + f"\n\nQuestion: {ex['ambiguous_question']}",
                 "output": ann["long_answer"]})

with open(D / "rag_asqa.json", "w", encoding="utf-8") as f:
    json.dump(rag_asqa, f)

print(len(rag_asqa))