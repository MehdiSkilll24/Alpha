# prefetch.py
import json, time
from pathlib import Path
from datasets import load_dataset
from rag_local import web_retrieve

CACHE = Path("/content/drive/MyDrive/Transformers/Alpha/hits_cache.json") # must be the same path eval_qa.py uses
N = 1000
DATA_DIR = r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\Alpha\dataset"

cache = json.load(open(CACHE)) if CACHE.exists() else {}

ds = load_dataset(
    "parquet",
    data_files={
        "train": f"{DATA_DIR}\\train-00000-of-00001.parquet",
        "validation": f"{DATA_DIR}\\validation-00000-of-00001.parquet",
        "test": f"{DATA_DIR}\\test-00000-of-00001.parquet",
    }
)

train_ds = ds["train"]
val_ds = ds["validation"]
test_ds = ds["test"]

ds = val_ds.shuffle(seed=0).select(range(N)) 
failed = 0                                   
for i, ex in enumerate(ds):
    q = ex["question"].strip()
    if q in cache:
        print(f"{i+1}/{N} cached   | {q[:60]}")
    else:
        h = web_retrieve(q)
        if h:
            cache[q] = h
            print(f"{i+1}/{N} fetched  | {len(h)} hits | {q[:60]}")
        else:
            failed += 1                      
            print(f"{i+1}/{N} FAILED   | {q[:60]}")
        time.sleep(3)
    if (i + 1) % 25 == 0:
        json.dump(cache, open(CACHE, "w"))
        print(f"--- saved: {len(cache)} cached | {failed} failed so far ---")

json.dump(cache, open(CACHE, "w"))
print(f"done: {len(cache)} of {N} cached | {failed} failed this run")
        
json.dump(cache, open(CACHE, "w"))
print("done:", len(cache), "of", N, "cached")