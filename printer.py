# peek_data.py
import json, random
from pathlib import Path

DATA = Path("/content/drive/MyDrive/Transformers/Alpha/jsons")   # change if needed
FILES = ["search_ex", "calc_ex", "rag_ex", "rag_long", "rag_ms_sentence", "rag_asqa"]
random.seed()

def clip(s, n=250):
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n // 2] + " ... " + s[-n // 2:]

for name in FILES:
    p = DATA / f"{name}.json"
    if not p.exists():
        print(f"\n=== {name}: NOT FOUND ===")
        continue
    rows = json.load(open(p, encoding="utf-8"))
    instrs = {r["instruction"] for r in rows}
    avg = sum(len(r["output"].split()) for r in rows) / max(1, len(rows))
    print(f"\n=== {name}: {len(rows)} rows | avg answer {avg:.1f} words | {len(instrs)} distinct instruction(s) ===")
    for r in random.sample(rows, min(3, len(rows))):
        print("  instruction:", clip(r["instruction"]))
        if r.get("input"):
            print("  input      :", clip(r["input"]))
        print("  output     :", clip(r["output"]))
        print()