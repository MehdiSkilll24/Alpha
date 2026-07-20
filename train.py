import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from main import Transformer
from Dataset import TranslationDataset
from tokenizer import Tokenizer
from collate_fn import collate_fn
import os

LR = 3e-4
BATCH_SIZE = 16
EPOCHS = 6


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
    batch = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(
        "parquet",
        data_files=r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\Alpha\train-00000-of-00001.parquet"
    )

    #------------
    # train/test split
    #------------
    split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    test_data = split["test"]

    #------------
    # tokenizer (built ONLY from train_data)
    #------------
    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()

    src_tokenizer.build_vocab([sample["translation"]["en"] for sample in train_data])
    tgt_tokenizer.build_vocab([sample["translation"]["fr"] for sample in train_data])

    #------------
    # dataset/dataloader
    #------------
    train_dataset = TranslationDataset(train_data, src_tokenizer, tgt_tokenizer)
    test_dataset = TranslationDataset(test_data, src_tokenizer, tgt_tokenizer)

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    #------------
    #   MODEL
    #------------
    model = Transformer(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    #------------
    # TRAINING
    #------------
    start_epoch = 0
    start_batch_idx = 0

    if os.path.exists("checkpoint_latest.pt"):
        checkpoint = torch.load("checkpoint_latest.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_batch_idx = checkpoint["batch_idx"] + 1

        if start_batch_idx >= len(loader):
            start_epoch += 1
            start_batch_idx = 0

        print(f"Resuming from epoch {start_epoch+1}, batch {start_batch_idx}")

    quarter = len(loader) // 4

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch_idx, (src, tgt) in enumerate(loader):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_correct += correct
            total_tokens += total
            accuracy = total_correct / total_tokens
            batch += BATCH_SIZE

            if quarter > 0 and (batch_idx + 1) % quarter == 0:
                checkpoint_data = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "src_vocab": src_tokenizer.word_to_idx,
                    "tgt_vocab": tgt_tokenizer.word_to_idx,
                }
                quarter_num = (batch_idx + 1) // quarter
                torch.save(checkpoint_data, f"checkpoint_epoch{epoch+1}_q{quarter_num}.pt")
                torch.save(checkpoint_data, "checkpoint_latest.pt")

            print(
                f"Batch {batch} | "
                f"Epoch {epoch+1} | "
                f"Loss: {running_loss / len(loader):.4f} | "
                f"Accuracy: {accuracy:.2%}"
            )

        # end-of-epoch evaluation on held-out test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"[Eval] Epoch {epoch+1} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2%}")

if __name__ == "__main__":
    train()