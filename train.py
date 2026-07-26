import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
from main import Transformer
from collate_fn import collate_fn
from Dataset import TranslationDataset
import shutil
import zipfile
from torch.amp import autocast, GradScaler
import json
import time
from bpe_tokenizer import build_bpe_tokenizer

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")

with open(config_path) as f:
    config = json.load(f)

LR = 3e-4
BATCH_SIZE = 32
EPOCHS = 16

# Local Colab disk — fast, reliable writes
LOCAL_DIR = r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\Alpha\checkpoints"
# Drive — persistent storage, only written to AFTER a verified local save
DRIVE_DIR = "/content/drive/MyDrive/Colab Notebooks/Alpha"


os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(DRIVE_DIR, exist_ok=True)


def save_checkpoint_safely(checkpoint_data, filename):
    """Save locally first, verify it's a valid file, then copy to Drive.
    Never trust a save until it's been read back successfully."""
    local_path = os.path.join(LOCAL_DIR, filename)
    drive_path = os.path.join(DRIVE_DIR, filename)

    torch.save(checkpoint_data, local_path)

    try:
        with zipfile.ZipFile(local_path) as z:
            z.namelist()  # forces a real read, not just open
    except zipfile.BadZipFile:
        print(f"WARNING: {filename} failed integrity check after saving locally — NOT copying to Drive.")
        return False

    shutil.copy(local_path, drive_path)

    try:
        with zipfile.ZipFile(drive_path) as z:
            z.namelist()
    except zipfile.BadZipFile:
        print(f"WARNING: {filename} corrupted during copy to Drive — local copy still intact at {local_path}.")
        return False

    print(f"Checkpoint verified and saved: {filename}")
    return True


def load_checkpoint_safely(filename):
    """Try Drive first, fall back to local if Drive copy is bad."""
    if os.path.exists(filename):
        try:
            with zipfile.ZipFile(filename) as z:
                z.namelist()
            return torch.load(filename, map_location="cpu")
        except zipfile.BadZipFile:
            print(f"WARNING: {filename} is corrupted, trying next option...")
    return None


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]

            logits = model(src, decoder_input)

            predictions = logits.argmax(dim=-1)
            mask = target != 0
            correct = ((predictions == target) & mask).sum().item()
            total = mask.sum().item()

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
            )

            total_loss += loss.item()
            total_correct += correct
            total_tokens += total

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    model.train()
    return avg_loss, accuracy


def train():
    batch_count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = load_dataset(
        "parquet",
        data_files={

            "train":
            r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\opus100_en-fr_train.parquet",

            "validation":
            r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\opus100_en-fr_validation.parquet",

            "test":
            r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\opus100_en-fr_test.parquet"
        }
    )


    train_data = dataset["train"]
    test_data = dataset["test"]

    combined_texts = (
    [sample["translation"]["en"] for sample in train_data]
    + [sample["translation"]["fr"] for sample in train_data]
    )

    tokenizer = build_bpe_tokenizer(combined_texts, vocab_size=32000, save_path=os.path.join(script_dir, "bpe_tokenizer.json"))
    actual_vocab_size = len(tokenizer)
    print(f"Actual vocab size: {actual_vocab_size}")
    
    train_dataset = TranslationDataset(train_data, tokenizer)
    test_dataset = TranslationDataset(test_data, tokenizer)

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    model = Transformer(
        actual_vocab_size,
        actual_vocab_size,
        config["d_model"],
        config["num_heads"],
        config["d_ff"],
        config["num_encoder_layers"],
        config["num_decoder_layers"],
        model_type=config["model_type"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    #------------
    # TRAINING — fresh start, no checkpoint loading this run
    #------------
    start_epoch = 0
    start_batch_idx = 0

    checkpoint = load_checkpoint_safely(r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\Alpha\checkpoints\checkpoint_latest.pt")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_batch_idx = checkpoint["batch_idx"] + 1

        if start_batch_idx >= len(loader):
            start_epoch += 1
            start_batch_idx = 0

        print(f"Resuming from epoch {start_epoch+1}, batch {start_batch_idx}")
    else:
        print("No valid checkpoint found — starting fresh from epoch 1.")

    quarter = len(loader) // 4

    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch_idx, (src, tgt) in enumerate(loader):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]

            optimizer.zero_grad()

            with autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(src, decoder_input)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            predictions = logits.argmax(dim=-1)
            mask = target != 0
            correct = ((predictions == target) & mask).sum().item()
            total = mask.sum().item()

            total_correct += correct
            total_tokens += total
            accuracy = total_correct / total_tokens if total_tokens > 0 else 0
            batch_count += BATCH_SIZE

            if quarter > 0 and (batch_idx + 1) % quarter == 0:
                checkpoint_data = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "vocab": tokenizer.word_to_idx,
                }
                quarter_num = (batch_idx + 1) // quarter
                save_checkpoint_safely(checkpoint_data, f"checkpoint_epoch{epoch+1}_q{quarter_num}.pt")
                save_checkpoint_safely(checkpoint_data, "checkpoint_latest.pt")

            print(
                f"Batch {batch_idx + 1} | "
                f"Epoch {epoch+1} | "
                f"Loss: {running_loss / (batch_idx + 1):.4f} | "
                f"Accuracy: {accuracy:.2%}"
            )

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"[Eval] Epoch {epoch+1} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2%}")
        scheduler.step(test_loss)
        print(optimizer.param_groups[0]['lr'])

        epoch_elapsed = time.time() - epoch_start_time  
        hours, rem = divmod(epoch_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        epoch_time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"  

        train_loss_last = running_loss / len(loader)
        train_acc_last = total_correct / total_tokens if total_tokens > 0 else 0

        results_line = (
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss_last:.4f} | Train Accuracy: {train_acc_last:.2%} | "
            f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2%} | "
            f"Time: {epoch_time_str}\n"
        )

        local_results = os.path.join(LOCAL_DIR, "epoch_results.txt")
        with open(local_results, "a") as f:
            f.write(results_line)
        shutil.copy(local_results, os.path.join(DRIVE_DIR, "epoch_results.txt"))

if __name__ == "__main__":
    train()