# 1. Round-trip a real chunk of training data
import numpy as np
data = np.memmap("train.bin", dtype=np.uint16, mode="r")   # use your real dtype
chunk = data[1_000_000:1_000_100].astype(int).tolist()
text = tok.decode(chunk)
print(text)
print(tok.encode(text) == chunk)   # should be True (or very close)

# 2. Look at what your prompt actually becomes
ids = tok.encode("The capital of France is")
print(ids, [tok.idx_to_word[i] for i in ids])   # adjust the reverse-map name to yours