# make_rag_wiki.py
# Builds reader examples from Wikipedia: passages (lead paragraph + 2 distractors)
# + a "What/Who is X?" question -> the article's first sentence as the answer.
import json, random, re
from pathlib import Path
from datasets import load_dataset
from prompts import INSTR

random.seed(0)
ROOT = Path(__file__).parent.absolute().parent
JSON = ROOT / "jsons" 
N_SCAN, N_ROWS = 60000, 2000

# sentence ends in an abbreviation or a single initial -> the splitter cut it too early
ABBR = re.compile(r"\b(ca|c|approx|vs|No|St|Dr|Mr|Mrs|Ms|Jr|Sr|Inc|Ltd|Co)\.$|\b[A-Za-z]\.$")
# "(born ...)", "(died ...)", or a year range in brackets -> probably a person
PERSON = re.compile(r"\([^)]*(\bborn\b|\bdied\b|\bb\.|\bd\.)"
                    r"|\([^)]*\b(1[0-9]{3}|20[0-2][0-9])\s*[–-]\s*(\w+\s+)*(1[0-9]{3}|20[0-2][0-9])")

Q_THING = ["What is {t}?", "What's {t}?", "What is {t}", "Define {t}.", "Tell me what {t} is."]
Q_PERSON = ["Who is {t}?", "Who was {t}?", "Who is {t}", "Tell me about {t}."]

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

arts = []
for i, a in enumerate(ds):
    if i == N_SCAN:
        break
    t, text = a["title"], a["text"]
    if any(c.isdigit() for c in t) or "(" in t or "List" in t or len(text.split()) < 150:
        continue
    raw_first = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    first = re.sub(r"\s*\([^)]*\)", "", raw_first).strip()      # drop pronunciation/date brackets
    if ABBR.search(first) or "may refer to" in first or not first.endswith((".", "!", "?")):
        continue
    if not (8 <= len(first.split()) <= 45):
        continue
    if not re.search(r"\b(is|are|was|were)\b", " ".join(first.split()[:15])):
        continue
    para = " ".join(text.split()[:150])
    arts.append((t, para, first, bool(PERSON.search(raw_first))))

random.shuffle(arts)
rag_wiki = []
for t, para, first, person in arts[:N_ROWS]:
    others = random.sample(arts, 2)
    ctx = [para] + [o[1] for o in others]
    random.shuffle(ctx)
    q = random.choice(Q_PERSON if person else Q_THING).format(t=t)
    rag_wiki.append({"instruction": INSTR,
                 "input": "\n\n".join(ctx) + f"\n\nQuestion: {q}",
                 "output": first})

with open(JSON / "rag_wiki.json", "w", encoding="utf-8") as f:
    json.dump(rag_wiki, f)

print(len(rag_wiki), "rows ->", JSON)

qs = lambda r: r["input"].rsplit("Question:", 1)[-1].strip()
who = [r for r in rag_wiki if qs(r).lower().startswith(("who", "tell me about"))]
what = [r for r in rag_wiki if r not in who]
print(f"\n{len(who)} person-style, {len(what)} thing-style")
for name, group in (("WHO", who), ("WHAT", what)):
    print(f"\n--- {name} samples ---")
    for r in random.sample(group, min(6, len(group))):
        print("-", qs(r), "->", r["output"])