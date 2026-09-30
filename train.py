import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
from main import Transformer
from bpe import load_bpe_tokenizer
from Dataset import PackedBinDataset
import shutil
import zipfile
from torch.amp import autocast, GradScaler
import json
import numpy as np
import random
import time
import math

LR = 3e-4
MIN_LR = LR * 0.1
MICRO_BATCH = 8
GRAD_ACCUM = 8
SEQ_LEN = 2048
MAX_STEPS = 10_000      
WARMUP_STEPS = 30
EVAL_EVERY = 1300
EVAL_BATCHES = 200
SAVE_EVERY = 300
LOG_EVERY = 10
SEED = 42

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Call at start of train.py
set_seed(42)

SCRIPT_DIR = Path(__file__).parent.absolute()
CONFIG_PATH = SCRIPT_DIR / "config.json"


# Determine environment and set checkpoint paths
if os.path.exists("/content/drive"):  # Colab environment
    LOCAL_DIR = Path("/content/Alpha/checkpoints")
    DRIVE_DIR = Path("/content/drive/MyDrive/Colab Notebooks/Alpha")
    print("Running in Colab — using Google Drive for persistent storage")
else:  # Local machine
    LOCAL_DIR = SCRIPT_DIR / "checkpoints"
    DRIVE_DIR = SCRIPT_DIR / "checkpoints"  # Same as local for non-Colab
    print(f"Running locally — using {LOCAL_DIR} for checkpoints")
 
# Create directories
os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(DRIVE_DIR, exist_ok=True)
 
print(f"Script directory: {SCRIPT_DIR}")
print(f"Config path: {CONFIG_PATH}")
print(f"Local checkpoint dir: {LOCAL_DIR}")
print(f"Drive checkpoint dir: {DRIVE_DIR}\n")
 
# Load config
with open(CONFIG_PATH) as f:
    config = json.load(f)
 
 
def save_checkpoint_safely(checkpoint_data, filename):
    """Save locally first, verify it's a valid file, then copy to Drive.
    Never trust a save until it's been read back successfully."""
    local_path = LOCAL_DIR / filename
    drive_path = DRIVE_DIR / filename
 
    torch.save(checkpoint_data, str(local_path))
 
    try:
        with zipfile.ZipFile(str(local_path)) as z:
            z.namelist()  # forces a real read, not just open
    except zipfile.BadZipFile:
        print(f"WARNING: {filename} failed integrity check after saving locally — NOT copying to Drive.")
        return False
 
    if LOCAL_DIR.resolve() != DRIVE_DIR.resolve():
        shutil.copy(str(local_path), str(drive_path))

 
    try:
        with zipfile.ZipFile(str(drive_path)) as z:
            z.namelist()
    except zipfile.BadZipFile:
        print(f"WARNING: {filename} corrupted during copy to Drive — local copy still intact at {local_path}.")
        return False
 
    print(f"Checkpoint verified and saved: {filename}")
    return True
 
 
def load_checkpoint_safely(filename):
    for d in (LOCAL_DIR, DRIVE_DIR):
        path = d / filename
        if not path.exists():
            continue
        try:
            with zipfile.ZipFile(str(path)) as z:
                z.namelist()
            print(f"Loading checkpoint from {path}")
            return torch.load(str(path), map_location="cpu")
        except zipfile.BadZipFile:
            print(f"WARNING: {path} is corrupted, trying next location...")
    return None


def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(progress, 1.0)))


class StepSampler:
    """Deterministic: the same step always yields the same sequence indices."""
    def __init__(self, n, per_step, seed):
        self.n, self.per_step, self.seed = n, per_step, seed
        self._epoch, self._perm = -1, None

    def indices(self, step):
        out = []
        for pos in range(step * self.per_step, (step + 1) * self.per_step):
            epoch, off = divmod(pos, self.n)
            if epoch != self._epoch:
                self._perm = np.random.default_rng([self.seed, epoch]).permutation(self.n)
                self._epoch = epoch
            out.append(int(self._perm[off]))
        return out


@torch.no_grad()
def evaluate(model, dataset, criterion, device, max_batches):
    model.eval()
    total_loss, total_correct, total_tokens, n = 0.0, 0, 0, 0
    for b in range(max_batches):
        start = b * MICRO_BATCH
        if start + MICRO_BATCH > len(dataset):
            break
        tokens = torch.stack([dataset[i] for i in range(start, start + MICRO_BATCH)]).to(device)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        with autocast("cuda", enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_loss += loss.item(); n += 1
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total_tokens += targets.numel()
    model.train()
    return total_loss / max(n, 1), total_correct / max(total_tokens, 1)


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_bpe_tokenizer(SCRIPT_DIR / "fineweb_bpe.json")

    PAD_ID = tokenizer.word_to_idx["<pad>"]
    config.update(
        pad_token_id=PAD_ID,
        unk_token_id=tokenizer.word_to_idx["<unk>"],
        bos_token_id=tokenizer.word_to_idx["<bos>"],
        eos_token_id=tokenizer.word_to_idx["<eos>"],
        vocab_size=len(tokenizer),
    )
    print(f"Vocab size: {len(tokenizer)}")

    train_ds = PackedBinDataset(SCRIPT_DIR / "train.bin", seq_len=SEQ_LEN)
    val_ds = PackedBinDataset(SCRIPT_DIR / "val.bin", seq_len=SEQ_LEN)
    sampler = StepSampler(len(train_ds), MICRO_BATCH * GRAD_ACCUM, SEED)

    model = Transformer(
        config["d_model"], config["num_heads"], config["vocab_size"],
        config["d_ff"], config["num_kv_heads"], config["num_decoder_layers"],
        dropout=config["Dropout"],
    ).to(device)


    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95),
    )

    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    arch = {k: config[k] for k in
            ("d_model", "num_heads", "num_kv_heads", "d_ff", "num_decoder_layers", "vocab_size")}
    start_step, best_val = 0, float("inf")

    ckpt = load_checkpoint_safely("checkpoint_latest.pt")
    if ckpt is not None:
        if ckpt.get("arch") != arch:
            raise RuntimeError(f"Checkpoint arch {ckpt.get('arch')} != current {arch}. "
                               "Rename or delete the old checkpoint.")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step, best_val = ckpt["step"], ckpt["best_val"]
        print(f"Resuming at step {start_step}")
    else:
        print("No checkpoint found, starting fresh at step 0.")

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    config["num_parameters"] = sum(p.numel() for p in model.parameters())

    def make_ckpt(next_step):
        return {"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "step": next_step, "best_val": best_val, "arch": arch}

    model.train()
    t0, tokens_since = time.perf_counter(), 0
    tokens_per_step = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN

    for step in range(start_step, MAX_STEPS):
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        idx = sampler.indices(step)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for m in range(GRAD_ACCUM):
            chunk = idx[m * MICRO_BATCH:(m + 1) * MICRO_BATCH]
            tokens = torch.stack([train_ds[i] for i in chunk]).to(device, non_blocking=True)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            scaler.scale(loss / GRAD_ACCUM).backward()
            step_loss += loss.item() / GRAD_ACCUM

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        tokens_since += tokens_per_step

        if (step + 1) % LOG_EVERY == 0:
            dt = time.perf_counter() - t0
            print(f"step {step+1}/{MAX_STEPS} | loss {step_loss:.4f} | lr {lr:.2e} | "
                  f"gnorm {grad_norm:.2f} | {tokens_since/dt:,.0f} tok/s")
            t0, tokens_since = time.perf_counter(), 0

        if (step + 1) % EVAL_EVERY == 0 or step + 1 == MAX_STEPS:
            val_loss, val_acc = evaluate(model, val_ds, criterion, device, EVAL_BATCHES)
            print(f"[Eval] step {step+1} | val loss {val_loss:.4f} | "
                  f"ppl {math.exp(val_loss):.2f} | acc {val_acc:.2%}")
            with open(LOCAL_DIR / "results.txt", "a") as f:
                f.write(f"step {step+1} | train {step_loss:.4f} | val {val_loss:.4f} | "
                        f"ppl {math.exp(val_loss):.2f} | acc {val_acc:.2%}\n")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint_safely({"model_state_dict": model.state_dict(), "arch": arch,
                                        "step": step + 1, "val_loss": val_loss},
                                       "checkpoint_best.pt")

        if (step + 1) % SAVE_EVERY == 0 or step + 1 == MAX_STEPS:
            save_checkpoint_safely(make_ckpt(step + 1), "checkpoint_latest.pt")
            if LOCAL_DIR.resolve() != DRIVE_DIR.resolve():
                shutil.copy(LOCAL_DIR / "results.txt", DRIVE_DIR / "results.txt")

if __name__ == "__main__":
    train()