"""Build dev/test routing-eval sets for Alpha's search / calc / answer router.

Sources (all NOT in SFT data; Alpaca is deliberately avoided because I trained on it):
  search : TriviaQA + NQ-Open questions, templated "current events" prompts, hand prompts
  calc   : programmatic arithmetic in many surface forms (with expected answers) + hand prompts
  answer : Dolly-15k (brainstorming / creative / classification / summarization / extraction /
           closed-book-with-passage), identity + chat phrases, hand prompts

Everything is deduped, and checked for near-duplicates against SFT prompt files.

Usage (from the model/ folder; needs `pip install datasets`):
  python build_routing_eval.py --hand routing_prompts.json \
      --train-files ../jsons/search_ex.json ../jsons/calc_ex.json ../jsons/chat.json \
                    ../jsons/identity.json ../jsons/rag_ex.json ../jsons/rag_long.json \
                    ../jsons/rag_asqa.json ../jsons/rag_wiki.json
Outputs: routing_dev.json, routing_test.json, routing_audit.json
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["search", "calc", "answer"]


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def nwords(s):
    return len(s.split())


# --------------------------------------------------------- leakage index
def instruction_of(ex):
    if "messages" in ex:
        return next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
    return ex.get("instruction", "")


def load_train_texts(paths):
    texts = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            texts += [instruction_of(ex) for ex in json.load(f)]
    return [t for t in texts if t]


class LeakIndex:
    """Exact + word-Jaccard near-duplicate check against training prompts.
    Very common words are skipped in the index, so intersections are slightly
    underestimated (it can miss a borderline duplicate, never invent one)."""

    def __init__(self, texts, thr=0.7, skip_df=2000):
        self.thr = thr
        self.sets = [words(t) for t in texts]
        self.exact = {norm(t) for t in texts}
        self.post = defaultdict(list)
        for i, s in enumerate(self.sets):
            for w in s:
                self.post[w].append(i)
        self.skip = {w for w, p in self.post.items() if len(p) > skip_df}
        print(f"Leak index: {len(texts)} training prompts")

    def hit(self, text):
        if norm(text) in self.exact:
            return True
        w = words(text)
        if not w:
            return False
        cnt = Counter()
        for x in w:
            if x in self.skip:
                continue
            for i in self.post.get(x, ()):
                cnt[i] += 1
        for i, inter in cnt.items():
            if inter / (len(w) + len(self.sets[i]) - inter) >= self.thr:
                return True
        return False


# ------------------------------------------------------------ calc generators
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
       "*": lambda a, b: a * b, "/": lambda a, b: a / b}
SYMS = {"+": ["+"], "-": ["-"], "*": ["*", "×", "x"], "/": ["/", "÷"]}
WORDS = {"+": ["plus"], "-": ["minus"], "*": ["times", "multiplied by"], "/": ["divided by"]}
ITEMS = ["pencils", "apples", "notebooks", "tickets", "stamps", "cookies", "oranges", "books"]


def fmt(v):
    v = round(v, 4)
    return str(int(v)) if float(v).is_integer() else f"{v:.4f}".rstrip("0").rstrip(".")


def rnum(rng):
    r = rng.random()
    if r < 0.6:
        return rng.randint(2, 99)
    if r < 0.9:
        return rng.randint(100, 9999)
    return round(rng.uniform(1, 99), 1)


def pair(rng, op):
    if op == "/":
        b, q = rng.randint(2, 25), rng.randint(2, 60)
        return b * q, b
    a, b = rnum(rng), rnum(rng)
    if op == "-" and a < b:
        a, b = b, a
    return a, b


def ev(op, a, b):
    return fmt(OPS[op](a, b))


def c_symbolic(rng):
    op = rng.choice(list(OPS))
    a, b = pair(rng, op)
    s = rng.choice(SYMS[op])
    return rng.choice([f"{a} {s} {b}", f"{a}{s}{b}", f"{a} {s} {b} =", f"{a} {s} {b} = ?"]), ev(op, a, b)


def c_verb(rng):
    op = rng.choice(list(OPS))
    a, b = pair(rng, op)
    e = f"{a} {rng.choice(SYMS[op])} {b}"
    t = rng.choice(["Calculate {}.", "Compute {}.", "Evaluate {}.", "Solve {}.",
                    "Work out {}.", "Please calculate {} for me.", "calculate {}"])
    return t.format(e), ev(op, a, b)


def c_wh(rng):
    op = rng.choice(list(OPS))
    a, b = pair(rng, op)
    e = f"{a} {rng.choice(WORDS[op])} {b}"
    t = rng.choice(["What is {}?", "What's {}?", "How much is {}?",
                    "Can you tell me what {} is?", "what is {}", "Tell me {}."])
    return t.format(e), ev(op, a, b)


def c_named(rng):
    t = rng.choice(["add", "sub", "mul", "div", "sum", "prod", "diff", "quot"])
    op = {"add": "+", "sum": "+", "sub": "-", "diff": "-",
          "mul": "*", "prod": "*", "div": "/", "quot": "/"}[t]
    a, b = pair(rng, op)
    p = {"add": f"Add {a} and {b}.", "sub": f"Subtract {b} from {a}.",
         "mul": f"Multiply {a} by {b}.", "div": f"Divide {a} by {b}.",
         "sum": f"Find the sum of {a} and {b}.", "prod": f"What is the product of {a} and {b}?",
         "diff": f"What is the difference between {a} and {b}?",
         "quot": f"Find the quotient of {a} and {b}."}[t]
    return p, ev(op, a, b)


def c_percent(rng):
    p = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 75])
    n = rng.choice([20, 40, 45, 60, 80, 120, 150, 200, 240, 250, 360, 500])
    k = rng.randrange(4)
    if k == 0:
        return f"What is {p}% of {n}?", fmt(n * p / 100)
    if k == 1:
        return f"Calculate {p} percent of {n}.", fmt(n * p / 100)
    if k == 2:
        return f"What is a {p}% tip on ${n}?", fmt(n * p / 100)
    return f"What is the sale price of a ${n} item after a {p}% discount?", fmt(n * (1 - p / 100))


def c_power(rng):
    k = rng.randrange(5)
    if k == 0:
        a = rng.randint(2, 30)
        return f"What is {a} squared?", fmt(a * a)
    if k == 1:
        a = rng.randint(2, 12)
        return f"What is {a} cubed?", fmt(a ** 3)
    a, b = rng.randint(2, 9), rng.randint(2, 5)
    if k == 2:
        return f"What is {a} to the power of {b}?", fmt(a ** b)
    if k == 3:
        return f"Evaluate {a}^{b}.", fmt(a ** b)
    r = rng.randint(2, 30)
    return f"What is the square root of {r * r}?", fmt(r)


def c_multi(rng):
    a, b, c = (rng.randint(2, 30) for _ in range(3))
    kinds = [(f"({a} + {b}) * {c}", (a + b) * c), (f"{a} + {b} * {c}", a + b * c),
             (f"{a} * {b} - {c}", a * b - c), (f"({a} + {b}) - {c}", a + b - c),
             (f"{a} * {b} + {c}", a * b + c), (f"{a} + {b} + {c}", a + b + c)]
    e, v = rng.choice(kinds)
    t = rng.choice(["Compute {}.", "What is {}?", "{}", "Evaluate the expression {}.", "calc {}"])
    return t.format(e), fmt(v)


def c_word(rng):
    k = rng.randrange(7)
    a, b, p = rng.randint(3, 40), rng.randint(2, 20), rng.randint(2, 25)
    item = rng.choice(ITEMS)
    if k == 0:
        return f"If I buy {a} {item} at ${p} each, how much do I pay in total?", fmt(a * p)
    if k == 1:
        s, t = rng.choice([40, 50, 60, 80, 90, 100, 120]), rng.choice([2, 3, 4, 5, 6])
        return f"A car travels at {s} km per hour for {t} hours. How far does it go?", fmt(s * t)
    if k == 2:
        return f"There are {a} students in a class and {b} more join. How many students are there now?", fmt(a + b)
    if k == 3:
        return f"A box holds {a} {item}. How many {item} are in {b} boxes?", fmt(a * b)
    if k == 4:
        x = a + b + rng.randint(1, 50)
        return f"Sara had ${x} and spent ${a}. How much money does she have left?", fmt(x - a)
    if k == 5:
        n, q = rng.randint(2, 12), rng.randint(2, 30)
        return f"{n * q} {item} are shared equally among {n} people. How many does each person get?", fmt(q)
    return f"A recipe needs {a} cups of flour per batch. How many cups are needed for {b} batches?", fmt(a * b)


def c_unit(rng):
    f, src, dst = rng.choice([(60, "hours", "minutes"), (60, "minutes", "seconds"), (24, "days", "hours"),
                              (100, "meters", "centimeters"), (1000, "kilograms", "grams"),
                              (7, "weeks", "days"), (12, "feet", "inches")])
    n = rng.randint(2, 40)
    return rng.choice([f"How many {dst} are in {n} {src}?", f"Convert {n} {src} to {dst}."]), fmt(n * f)


CALC_CATS = {  # name: (generator, share of the calc class)
    "symbolic": (c_symbolic, 0.12), "verb": (c_verb, 0.12), "wh_math": (c_wh, 0.14),
    "named_op": (c_named, 0.10), "percent": (c_percent, 0.12), "power_root": (c_power, 0.08),
    "multi_step": (c_multi, 0.10), "word_problem": (c_word, 0.14), "unit_conv": (c_unit, 0.08),
}


def calc_items(rng, fn, cat, n):
    out, seen = [], set()
    for _ in range(n):
        p, e = fn(rng)
        if norm(p) not in seen:
            seen.add(norm(p))
            out.append({"prompt": p, "label": "calc", "category": cat, "source": "template", "expected": e})
    return out


# -------------------------------------------------- search / answer templates
def current_items(rng):
    roles = [("president", "France"), ("prime minister", "the United Kingdom"), ("prime minister", "Japan"),
             ("chancellor", "Germany"), ("president", "Brazil"), ("president", "South Korea"),
             ("prime minister", "Australia"), ("president", "Mexico"), ("prime minister", "India"),
             ("president", "Argentina")]
    cities = ["Paris", "Tokyo", "Cairo", "Toronto", "Berlin", "Sydney", "Lagos", "Madrid", "Seoul", "Lima"]
    things = ["gold", "silver", "oil", "bitcoin", "copper", "natural gas"]
    products = ["iPhone model", "Android version", "Windows version", "Tesla model", "PlayStation console"]
    events = ["World Cup", "Super Bowl", "Wimbledon final", "Tour de France", "Nobel Peace Prize", "Champions League final"]
    teams = ["Lakers", "Real Madrid", "Yankees", "Arsenal", "Barcelona"]
    ps = [f"Who is the current {r} of {c}?" for r, c in roles]
    ps += [f"What is the weather like in {c} today?" for c in cities]
    ps += [f"What is the current price of {t}?" for t in things]
    ps += [f"What is the latest {p}?" for p in products]
    ps += [f"Who won the most recent {e}?" for e in events]
    ps += [f"What was the score of the latest {t} game?" for t in teams]
    ps += ["What are today's top news headlines?", "What is the exchange rate between the dollar and the euro right now?"]
    rng.shuffle(ps)
    return [{"prompt": p, "label": "search", "category": "current", "source": "template"} for p in ps]


IDENTITY = ["What is your name?", "Who are you?", "Who made you?", "Who created you?", "Who built you?",
            "What are you?", "Are you a human or a bot?", "Tell me about yourself.",
            "What can you help me with?", "What can you do?", "What tools do you have?",
            "Which tools can you use?", "Introduce yourself.", "What is your purpose?", "Are you an AI?",
            "How do you work?", "Can you browse the internet?", "What should I call you?",
            "Who developed you?", "What kind of assistant are you?"]
CHAT = ["Hi!", "Hello there.", "Hey, how's it going?", "Good morning!", "Good evening!", "How are you?",
        "How are you doing today?", "Thanks!", "Thank you so much.", "Thanks, that helped.", "Bye!",
        "See you later.", "Goodnight.", "I'm bored.", "Can you tell me a joke?", "Tell me something funny.",
        "I had a rough day.", "Let's chat.", "What's up?", "Nice to meet you."]

DOLLY_CATS = {"brainstorming": 0.20, "creative_writing": 0.20, "classification": 0.15,
              "summarization": 0.15, "information_extraction": 0.15, "closed_qa": 0.15}


# ---------------------------------------------------------------- loaders
def load_trivia(rng, k=4000):
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
    out = []
    for ex in ds:
        q = ex["question"].strip()
        if 3 <= nwords(q) <= 30 and "\n" not in q:
            out.append(q)
        if len(out) >= k:
            break
    return out


def load_nq(rng, k=4000):
    from datasets import load_dataset
    try:
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
    except Exception:
        ds = load_dataset("nq_open", split="validation")
    qs = [ex["question"].strip() for ex in ds]
    rng.shuffle(qs)
    # NQ is lowercase with no '?': keep half raw (how people really type), half cleaned
    return [q if rng.random() < 0.5 else q[:1].upper() + q[1:] + "?" for q in qs[:k]]


def load_dolly():
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    pools = defaultdict(list)
    for ex in ds:
        c = ex["category"]
        if c not in DOLLY_CATS:
            continue
        ins, ctx = ex["instruction"].strip(), (ex["context"] or "").strip()
        p = f"{ins}\n\n{ctx}" if ctx else ins
        if 3 <= nwords(p) <= 150:
            pools[c].append(p)
    return pools


def safe(fn, name, *a):
    try:
        return fn(*a)
    except Exception as e:  # network / dataset-id problems should not kill the whole build
        print(f"  WARNING: could not load {name}: {type(e).__name__}: {e}")
        return None


# ------------------------------------------------------------- assembly
def allocate(pools, quotas, total):
    take = {s: min(quotas.get(s, 0), len(p)) for s, p in pools.items()}
    short = total - sum(take.values())
    while short > 0:
        spare = [s for s, p in pools.items() if len(p) > take[s]]
        if not spare:
            break
        for s in spare:
            if short == 0:
                break
            take[s] += 1
            short -= 1
    out = []
    for s, p in pools.items():
        out += p[:take[s]]
    return out, short


def split_shares(total, shares):
    q = {k: int(round(total * v)) for k, v in shares.items()}
    q[max(shares, key=shares.get)] += total - sum(q.values())
    return q


def build(args):
    rng = random.Random(args.seed)
    T = args.per_class
    leak = LeakIndex(load_train_texts(args.train_files), args.thr) if args.train_files else None
    seen = set()

    def prepare(items, limit, tag):
        rng.shuffle(items)
        kept, dup, flagged = [], 0, 0
        for it in items:
            n = norm(it["prompt"])
            if not n or n in seen:
                dup += 1
                continue
            if leak and leak.hit(it["prompt"]):
                flagged += 1
                continue
            seen.add(n)
            kept.append(it)
            if len(kept) >= limit:
                break
        print(f"  {tag:32} kept {len(kept):4} | duplicates {dup:4} | near-train {flagged:4}")
        return kept

    def plain(prompts, label, category, source):
        return [{"prompt": p, "label": label, "category": category, "source": source} for p in prompts]

    hand_all = []
    if args.hand and Path(args.hand).exists():
        with open(args.hand, encoding="utf-8") as f:
            hand_all = [{**x, "source": "hand"} for x in json.load(f)]
    print(f"Hand prompts: {len(hand_all)}")

    print("\nsearch")
    hand = prepare([x for x in hand_all if x["label"] == "search"], 10 ** 6, "search/hand")
    tpl = prepare(current_items(rng), 80, "search/current_tpl")
    rest = max(T - len(hand) - 30, 0)
    trivia = safe(load_trivia, "TriviaQA", rng) or []
    nq = safe(load_nq, "NQ-Open", rng) or []
    pools = {"hand": hand, "current_tpl": tpl,
             "trivia_qa": prepare(plain(trivia, "search", "trivia", "trivia_qa"), 2 * rest // 2 + 40, "search/trivia_qa"),
             "nq_open": prepare(plain(nq, "search", "nq", "nq_open"), 2 * rest // 2 + 40, "search/nq_open")}
    quotas = {"hand": len(hand), "current_tpl": 30, "trivia_qa": rest // 2, "nq_open": rest - rest // 2}
    search, short = allocate(pools, quotas, T)
    print(f"  -> {len(search)} search (short by {short})")

    print("\ncalc")
    hand = prepare([x for x in hand_all if x["label"] == "calc"], 10 ** 6, "calc/hand")
    rest = max(T - len(hand), 0)
    qs = split_shares(rest, {k: v[1] for k, v in CALC_CATS.items()})
    pools, quotas = {"hand": hand}, {"hand": len(hand)}
    for cat, (fn, _) in CALC_CATS.items():
        pools[cat] = prepare(calc_items(rng, fn, cat, qs[cat] * 4 + 20), 2 * qs[cat] + 20, f"calc/{cat}")
        quotas[cat] = qs[cat]
    calc, short = allocate(pools, quotas, T)
    print(f"  -> {len(calc)} calc (short by {short})")

    print("\nanswer")
    hand = prepare([x for x in hand_all if x["label"] == "answer"], 10 ** 6, "answer/hand")
    rest = max(T - len(hand) - 40, 0)
    dolly = safe(load_dolly, "Dolly-15k") or {}
    qs = split_shares(rest, DOLLY_CATS)
    pools = {"hand": hand,
             "identity": prepare(plain(IDENTITY, "answer", "identity", "template"), 40, "answer/identity"),
             "chat": prepare(plain(CHAT, "answer", "chat", "template"), 40, "answer/chat")}
    quotas = {"hand": len(hand), "identity": 20, "chat": 20}
    for cat in DOLLY_CATS:
        pools[f"dolly_{cat}"] = prepare(plain(dolly.get(cat, []), "answer", cat, "dolly"),
                                        2 * qs[cat] + 20, f"answer/dolly_{cat}")
        quotas[f"dolly_{cat}"] = qs[cat]
    answer, short = allocate(pools, quotas, T)
    print(f"  -> {len(answer)} answer (short by {short})")

    items = search + calc + answer
    if len(items) < args.min_total:
        raise SystemExit(f"\nOnly {len(items)} examples (< {args.min_total}). Check the WARNING lines above "
                         "(dataset downloads failed?) or lower --train-files strictness / raise --thr.")
    return items, rng


def split(items, rng):
    groups = defaultdict(list)
    for it in items:
        groups[(it["label"], it["source"], it["category"])].append(it)
    dev, test = [], []
    for k in sorted(groups):
        g = groups[k]
        rng.shuffle(g)
        off = rng.randrange(2)
        for i, it in enumerate(g):
            (dev if (i + off) % 2 == 0 else test).append(it)
    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="routing_prompts.json")
    ap.add_argument("--train-files", nargs="*", default=[])
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--min-total", type=int, default=750)
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    items, rng = build(args)
    dev, test = split(items, rng)
    audit = rng.sample(items, max(60, len(items) // 10))

    out = Path(args.outdir)
    for name, data in [("routing_dev.json", dev), ("routing_test.json", test), ("routing_audit.json", audit)]:
        with open(out / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)

    print(f"\nTotal {len(items)} | dev {len(dev)} | test {len(test)} | audit sample {len(audit)}")
    print("\nlabel x source:")
    for (lab, src), n in sorted(Counter((i["label"], i["source"]) for i in items).items()):
        print(f"  {lab:7} {src:10} {n}")
    for nm, d in [("dev", dev), ("test", test)]:
        print(f"{nm}: " + ", ".join(f"{l}={sum(i['label'] == l for i in d)}" for l in LABELS))


if __name__ == "__main__":
    main()