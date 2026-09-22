import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
from main import Transformer
from collate_fn import collate_fn
from bpe import build_bpe_tokenizer
from Dataset import WikiTextDataset
import shutil
import zipfile
from torch.amp import autocast, GradScaler
import json

with open("config.json") as f:
    config = json.load(f)

LR = 3e-4
BATCH_SIZE = 32
EPOCHS = 18

# Local Colab disk — fast, reliable writes
LOCAL_DIR = r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\Alpha\checkpoints"
# Drive — persistent storage, only written to AFTER a verified local save
DRIVE_DIR = "/content/drive/MyDrive/Colab Notebooks/Alpha"

with open("config.json") as f:
    config = json.load(f)

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
    if os.path.exists(filename):
        try:
            with zipfile.ZipFile(filename) as z:
                z.namelist()
            return torch.load(filename, map_location="cpu")
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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = load_dataset("wikitext", "wikitext-103-v1")

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    combined_texts = dataset["train"]["text"]
    tokenizer = build_bpe_tokenizer(combined_texts, vocab_size=32000)

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

    train_dataset = WikiTextDataset(train_data, tokenizer)
    test_dataset = WikiTextDataset(test_data, tokenizer)
    val_dataset = WikiTextDataset(val_data, tokenizer)

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn(PAD_ID),
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn(PAD_ID),
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
           val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn(PAD_ID),
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

    quarter = len(loader) // 4
    x = sum(p.numel() for p in model.parameters())
    print(x)
    for epoch in range(start_epoch, EPOCHS):
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
            scaler.step(optimizer)
            scaler.update()

            num_batches_processed += 1
            running_loss += loss.item()

            predictions = logits.argmax(dim=-1)
            mask = targets != PAD_ID
            correct = ((predictions == targets) & mask).sum().item()
            total = mask.sum().item()

            total_correct += correct
            total_tokens += total
            accuracy = total_correct / total_tokens if total_tokens > 0 else 0

            if quarter > 0 and (batch_idx + 1) % quarter == 0:
                checkpoint_data = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
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
                f"Loss: {running_loss / (num_batches_processed):.4f} | "
                f"Accuracy: {accuracy:.2%}"
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
            f"Perplexity: {perplexity:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_Acc:.2%}\n"
        )

        local_results = os.path.join(LOCAL_DIR, "epoch_results.txt")
        with open(local_results, "a") as f:
            f.write(results_line)
        shutil.copy(local_results, os.path.join(DRIVE_DIR, "epoch_results.txt"))

if __name__ == "__main__":
    train()