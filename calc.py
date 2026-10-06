import random, json, os
from pathlib import Path

D = Path("/content/drive/MyDrive/Transformers/Alpha") if os.path.exists("/content") else Path(__file__).parent.absolute()
random.seed(0)

ops = [("plus","+"),("minus","-"),("times","*"),("divided by","/")]
fm = ["What's {a} {w} {b}?", "Compute {a} {w} {b}", "Calculate {a} {w} {b}", "{a} {s} {b} = ?"]
calc_ex = []
for _ in range(3000):
    a, b = random.randint(2, 999), random.randint(2, 99)
    w, s = random.choice(ops)
    q = random.choice(fm).format(a=a, b=b, w=w, s=s)
    calc_ex.append({"instruction": q, "input": "", "output": f"[CALC] {a} {s} {b}"})

with open(D / "calc_ex.json", "w", encoding="utf-8") as f:
    json.dump(calc_ex, f)

print(len(calc_ex), "->", D / "calc_ex.json")