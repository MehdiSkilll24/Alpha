import operator
import re

CALC_RE = re.compile(r"\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*")
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}

def calc(expr):
    m = CALC_RE.fullmatch(expr)
    if not m:
        return None
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    if op == "/" and b == 0:
        return "Can't divide by zero."
    r = OPS[op](a, b)
    return str(int(r)) if r == int(r) else f"{r:.4f}".rstrip("0")