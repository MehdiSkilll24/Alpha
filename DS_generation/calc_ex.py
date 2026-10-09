# make_calc_data.py
import json, random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT = SCRIPT_DIR.parent
JSON = ROOT / "jsons"
JSON.mkdir(parents=True, exist_ok=True)
random.seed(0)

# canonical op -> surface forms
OPS = {
    "+": ["+", "plus"],
    "-": ["-", "minus"],
    "*": ["*", "x", "×", "times", "multiplied by"],
    "/": ["/", "÷", "divided by"],
}
SYMBOLS = {"+", "-", "*", "/", "x", "×", "÷"}

TRAIN_LEADS = ["", "", "what's ", "what is ", "whats ", "what ", "compute ",
               "calculate ", "calc ", "how much is ", "tell me ", "find "]
TEST_LEADS  = ["solve ", "evaluate ", "work out ", "what would be "]   # never seen in training
TAILS = ["", "", "?", " ?", " =", " = ?", "."]

def number(allow_neg=True):
    r = random.random()
    if r < 0.12:                                   # decimal
        return round(random.uniform(0.5, 200), random.choice([1, 2]))
    if r < 0.20 and allow_neg:                     # negative
        return -random.randint(1, 99)
    return random.randint(0, 999)

def fmt_num(x):
    return str(x)

def make(n, leads):
    rows = []
    for _ in range(n):
        a = number()
        b = number(allow_neg=False)
        if isinstance(b, float):
            b = round(abs(b), 1)
        if b == 0:
            b = random.randint(1, 99)              # never divide by zero
        op = random.choice(list(OPS))
        w = random.choice(OPS[op])
        # symbols can be tight ("5+5") or spaced; words always spaced
        sp = random.choice([" ", ""]) if w in SYMBOLS else " "
        q = random.choice(leads) + f"{fmt_num(a)}{sp}{w}{sp}{fmt_num(b)}" + random.choice(TAILS)
        q = random.choice([q, q, q.lower(), q.capitalize()])
        rows.append({"instruction": q.strip(), "input": "",
                     "output": f"[CALC] {fmt_num(a)} {op} {fmt_num(b)}"})
    return rows

calc_ex = make(8000, TRAIN_LEADS)
calc_test  = make(500, TEST_LEADS)

with open(JSON / "calc_ex.json", "w", encoding="utf-8") as f:
    json.dump(calc_ex, f)

with open(JSON / "calc_test.json", "w", encoding="utf-8") as f:
    json.dump(calc_test, f)
