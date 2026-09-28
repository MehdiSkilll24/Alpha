import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
from pathlib import Path
from main import Transformer
from bpe import build_bpe_tokenizer
from Dataset import PackedWikiTextDataset
import shutil
import zipfile
from torch.amp import autocast, GradScaler
import json
import numpy as np
import random
import time
LR = 3e-4
BATCH_SIZE = 64
EPOCHS = 30

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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
    local_path = LOCAL_DIR / filename
    if local_path.exists():
        try:
            with zipfile.ZipFile(str(local_path)) as z:
                z.namelist()
            return torch.load(str(local_path), map_location="cpu")
        except zipfile.BadZipFile:
            print(f"WARNING: {filename} is corrupted, trying next option...")
    return None


def evaluate(model, loader, criterion, device, pad_id):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for tokens in loader:
            tokens = tokens.to(device)
            inputs = tokens[:, :-1]

            targets = tokens[:, 1:]

            logits = model(inputs)

            predictions = logits.argmax(dim=-1)
            mask = targets != pad_id
            correct = ((predictions == targets) & mask).sum().item()
            total = mask.sum().item()

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            total_loss += loss.item()
            total_correct += correct
            total_tokens += total

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    model.train()
    return avg_loss, accuracy

def load_wikitext_safe():
    """Load WikiText with explicit error handling and retry logic"""
    from datasets import load_dataset
    
    print("Loading WikiText-103 dataset...")
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
            print("✓ Dataset loaded successfully\n")
            return dataset
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)[:100]}")
            if attempt < max_retries - 1:
                print("Retrying with cache flush...\n")
                # Clear HF cache and retry
                hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
                if hf_cache.exists():
                    try:
                        shutil.rmtree(hf_cache, ignore_errors=True)
                        print(f"Cleared cache at {hf_cache}\n")
                    except:
                        pass
            else:
                print(f"\n✗ Failed to load dataset after {max_retries} attempts")
                raise



def train():
    set_seed(42) # precaution reseed 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_wikitext_safe()
    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    combined_texts = dataset["train"]["text"]
    tokenizer = build_bpe_tokenizer(combined_texts, vocab_size=32000)
    #debug prints
    print(f"PAD_ID: {tokenizer.word_to_idx['<pad>']}")
    print(f"BOS_ID: {tokenizer.word_to_idx['<bos>']}")
    print(f"Vocab size: {len(tokenizer)}")

    PAD_ID = tokenizer.word_to_idx["<pad>"]
    BOS_ID = tokenizer.word_to_idx["<bos>"]
    EOS_ID = tokenizer.word_to_idx["<eos>"]
    UNK_ID = tokenizer.word_to_idx["<unk>"]
    actual_vocab_size = len(tokenizer)

    config["pad_token_id"] = PAD_ID
    config["unk_token_id"] = UNK_ID
    config["bos_token_id"] = BOS_ID
    config["eos_token_id"] = EOS_ID

    config["vocab_size"] = actual_vocab_size

    train_dataset = PackedWikiTextDataset(
    train_data,
    tokenizer,
    seq_len=1024
    )

    val_dataset = PackedWikiTextDataset(
        val_data,
        tokenizer,
        seq_len=1024
    )

    test_dataset = PackedWikiTextDataset(
        test_data,
        tokenizer,
        seq_len=1024
    )

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
           val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    model = Transformer(
        config["d_model"],
        config["num_heads"],
        config["vocab_size"],
        config["d_ff"],
        config["num_kv_heads"],
        config["num_decoder_layers"]
    ).to(device)
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    #------------
    # TRAINING — fresh start, no checkpoint loading this run
    #------------
    start_epoch = 0
    start_batch_idx = 0

    checkpoint = load_checkpoint_safely(r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\Alpha\checkpoints\checkpoint_latest.pt")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_batch_idx = checkpoint["batch_idx"] + 1

        if start_batch_idx >= len(loader):
            start_epoch += 1
            start_batch_idx = 0

        print(f"Resuming from epoch {start_epoch+1}, batch {start_batch_idx}")
    else:
        print("No valid checkpoint found — starting fresh from epoch 1.")

    total_params = sum(p.numel() for p in model.parameters())

    embedding_params = sum(
        p.numel() for p in model.embedding.parameters()
    )

    attention_params = sum(
        p.numel()
        for module in model.modules()
        if module.__class__.__name__ == "GQA"
        for p in module.parameters()
    )

    mlp_params = sum(
        p.numel()
        for module in model.modules()
        if module.__class__.__name__ == "MLP"
        for p in module.parameters()
    )

    norm_params = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, nn.LayerNorm)
        for p in module.parameters()
    )

    output_params = sum(
        p.numel()
        for p in model.output_proj.parameters()
    )

    print("\n===== MODEL PARAMETERS =====")
    print(f"Total:      {total_params:,}")
    print(f"Embedding:  {embedding_params:,}")
    print(f"Attention:  {attention_params:,}")
    print(f"MLP:        {mlp_params:,}")
    print(f"LayerNorm:  {norm_params:,}")
    print(f"Output:     {output_params:,}")
    print("============================\n")

    config["num_parameters"] = total_params

    cumulative_tokens = 0
    cumulative_effective_tokens = 0
    cumulative_time = 0.0

    for epoch in range(start_epoch, EPOCHS):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        epoch_start = time.perf_counter()
        epoch_processed_tokens = 0
        epoch_effective_tokens = 0
        model.train()
        running_loss = 0
        total_correct = 0
        total_tokens = 0
        num_batches_processed = 0

        for batch_idx, tokens in enumerate(loader):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

            tokens = tokens.to(device, non_blocking=True)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            
            optimizer.zero_grad()
            
            with autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(inputs)
            
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            num_batches_processed += 1
            running_loss += loss.item()

            predictions = logits.argmax(dim=-1)
            mask = targets != PAD_ID
            correct = ((predictions == targets) & mask).sum().item()
            total = mask.sum().item()
            # Tokens actually processed by the model
            processed_tokens = inputs.numel()

            # Non-padding tokens contributing to the loss
            effective_tokens = total

            epoch_processed_tokens += processed_tokens
            epoch_effective_tokens += effective_tokens

            total_correct += correct
            total_tokens += total
            accuracy = total_correct / total_tokens if total_tokens > 0 else 0


            print(
                f"Batch {batch_idx + 1} | "
                f"Epoch {epoch+1} | "
                f"Loss: {running_loss / (num_batches_processed):.4f} | "
                f"Accuracy: {accuracy:.2%}"
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - epoch_start

        cumulative_tokens += epoch_processed_tokens
        cumulative_effective_tokens += epoch_effective_tokens
        cumulative_time += epoch_time

        tokens_per_sec = epoch_processed_tokens / epoch_time
        effective_tokens_per_sec = epoch_effective_tokens / epoch_time

        cumulative_tokens_per_sec = cumulative_tokens / cumulative_time

        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            peak_vram = 0.0

        print(
            f"\n[Performance] Epoch {epoch+1} | "
            f"Time: {epoch_time:.2f}s | "
            f"Tokens/s: {tokens_per_sec:,.0f} | "
            f"Effective tokens/s: {effective_tokens_per_sec:,.0f} | "
            f"Peak VRAM: {peak_vram:.2f} GB"
        )

        val_loss, val_Acc = evaluate(model, val_loader, criterion, device, PAD_ID)
        perplexity = torch.exp(torch.tensor(val_loss))
        print(f"[Eval] Epoch {epoch+1} | Val loss: {val_loss:.4f} | Val Accuracy: {val_Acc:.2%}")
        print(optimizer.param_groups[0]['lr'])
        train_loss_last = running_loss / num_batches_processed
        train_acc_last = total_correct / total_tokens if total_tokens > 0 else 0

        results_line = (
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss_last:.4f} | "
            f"Train Accuracy: {train_acc_last:.2%} | "
            f"Val Perplexity: {perplexity:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_Acc:.2%} | "
            f"Epoch Time: {epoch_time:.2f}s | "
            f"Tokens/s: {tokens_per_sec:.0f} | "
            f"Effective Tokens/s: {effective_tokens_per_sec:.0f} | "
            f"Cumulative Tokens: {cumulative_tokens:,} | "
            f"Cumulative Time: {cumulative_time:.2f}s | "
            f"Peak VRAM: {peak_vram:.2f}GB\n"
        )

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "batch_idx": batch_idx,
            "vocab": tokenizer.word_to_idx,
        }
        save_checkpoint_safely(checkpoint_data, "checkpoint_latest.pt")

        local_results = os.path.join(LOCAL_DIR, "epoch_results.txt")
        with open(local_results, "a") as f:
            f.write(results_line)
        shutil.copy(local_results, os.path.join(DRIVE_DIR, "epoch_results.txt"))

if __name__ == "__main__":
    train()