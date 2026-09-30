import itertools, pickle
import numpy as np
from datasets import load_dataset
from bpe import build_bpe_tokenizer
from pathlib import Path
TARGET_TOKENS = 3_000_000_000   
VAL_DOCS = 5000                 # held out by document, not by slice
FLUSH = 5_000_000

def main():
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    texts = [d["text"] for d in itertools.islice(ds, 200_000)]
    OUT = Path(__file__).parent.absolute()
    tokenizer = build_bpe_tokenizer(texts, vocab_size=32000)

    assert len(tokenizer) < 65536          # uint16 limit

    val_f = open(OUT / "val.bin", "wb")
    train_f = open(OUT / "train.bin", "wb")
    buf, n_train = [], 0

    for i, doc in enumerate(ds):           # restarts the stream from doc 0
        ids = tokenizer.encode(doc["text"])
        if i < VAL_DOCS:
            val_f.write(np.array(ids, dtype=np.uint16).tobytes())
            continue
        buf.extend(ids)
        if len(buf) >= FLUSH:
            train_f.write(np.array(buf, dtype=np.uint16).tobytes())
            n_train += len(buf); buf = []
            print(f"{n_train:,} train tokens")
            if n_train >= TARGET_TOKENS:
                break
    train_f.close(); val_f.close()

if __name__ == "__main__":
    main()