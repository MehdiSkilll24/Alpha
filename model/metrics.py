import re, string
from collections import Counter

def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation)) #remove punctuation
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()) #fill the leftover space

def em(pred, golds):
    return max(norm(pred) == norm(g) for g in golds)

def f1(pred, golds):
    def one(p, g):
        p, g = norm(p).split(), norm(g).split()
        common = Counter(p) & Counter(g)
        n = sum(common.values()) #count how many predicted words match the correct ones
        if n == 0: return 0.0
        pr, rc = n / len(p), n / len(g)
        return 2 * pr * rc / (pr + rc) #formula
    return max(one(pred, g) for g in golds)

def contains(pred, golds):                 # fairer for sentence answers
    return max(norm(g) in norm(pred) for g in golds)