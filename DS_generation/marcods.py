# make_rag_long.py
import json, os, random
from pathlib import Path
from datasets import load_dataset
from prompts import INSTR
random.seed(0)

SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT = SCRIPT_DIR.parent
JSON = ROOT / "jsons"
ds = load_dataset("microsoft/ms_marco", "v2.1", split="train").shuffle(seed=42).select(range(4000))

rag_long = []
for ex in ds:
    ans = (ex["wellFormedAnswers"] or ex["answers"])
    ans = ans[0].strip() if len(ans) else ""
    if len(ans.split()) < 8 or "No Answer" in ans:        # keep real sentences only
        continue
    ps = ex["passages"]
    sel = [t for t, s in zip(ps["passage_text"], ps["is_selected"]) if s]
    oth = [t for t, s in zip(ps["passage_text"], ps["is_selected"]) if not s]
    pick = sel[:1] + random.sample(oth, min(2, len(oth)))
    random.shuffle(pick)                                   # answer passage not always first
    rag_long.append({"instruction": INSTR,
                "input": "\n\n".join(pick) + f"\n\nQuestion: {ex['query']}",
                "output": ans})
    if len(rag_long) == 4000: break


with open(JSON / "rag_long.json", "w", encoding="utf-8") as f:
    json.dump(rag_long, f)

print(len(rag_long))