from pathlib import Path
import json
import torch
from main import Transformer
import os

BASE = Path(__file__).parent
CONFIG_PATH = BASE / "config.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)

model = Transformer(
    config["d_model"], config["num_heads"], config["vocab_size"],
    config["d_ff"], config["num_kv_heads"], config["num_decoder_layers"],
    config["Dropout"],
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

config["num_parameters"] = n_params
tmp = CONFIG_PATH.with_suffix(".json.tmp")
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
os.replace(tmp, CONFIG_PATH)