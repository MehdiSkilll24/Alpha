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
import json
import numpy as np
import random

# ============================================================================
# SETUP - Match exactly with train.py
# ============================================================================

def set_seed(seed=42):
    """Match training reproducibility settings."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

set_seed(42)



SCRIPT_DIR = Path(__file__).parent.absolute()
CONFIG_PATH = SCRIPT_DIR / "config.json"
BATCH_SIZE = 32

if os.path.exists("/content/drive"):
    # Colab: fetch from Google Drive
    CHECKPOINT_DIR = Path("/content/drive/MyDrive/Colab Notebooks/Alpha")
else:
    # Local: fetch from Windows path
    CHECKPOINT_DIR = Path(r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\Alpha\checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Checkpoint dir: {CHECKPOINT_DIR}\n")

# Load config
with open(CONFIG_PATH) as f:
    config = json.load(f)


# ============================================================================
# HELPER FUNCTIONS - Copy from train.py
# ============================================================================

def load_checkpoint_safely(filename):
    """Load checkpoint with integrity check."""
    checkpoint_path = CHECKPOINT_DIR / filename
    if checkpoint_path.exists():
        try:
            with zipfile.ZipFile(str(checkpoint_path)) as z:
                z.namelist()
            print(f"✅ Loaded checkpoint: {filename}")
            return torch.load(str(checkpoint_path), map_location="cpu")
        except zipfile.BadZipFile:
            print(f"❌ WARNING: {filename} is corrupted")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
    return None


def load_wikitext_safe():
    """Load WikiText-103 dataset."""
    print("Loading WikiText-103 dataset...")
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
            print("✅ Dataset loaded successfully\n")
            return dataset
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)[:100]}")
            if attempt < max_retries - 1:
                print("Retrying with cache flush...\n")
                hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
                if hf_cache.exists():
                    try:
                        shutil.rmtree(hf_cache, ignore_errors=True)
                        print(f"Cleared cache at {hf_cache}\n")
                    except:
                        pass
            else:
                print(f"\n❌ Failed to load dataset after {max_retries} attempts")
                raise


def evaluate(model, loader, criterion, device, pad_id):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, tokens in enumerate(loader):
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
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}...")

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def eval_final():
    """
    Load final checkpoint and evaluate on test set.
    
    IMPORTANT: This evaluation should be identical to training's evaluation.
    - Same seed (deterministic)
    - Same batch size
    - NO SHUFFLING on test set
    - Same preprocessing
    """
    print("="*70)
    print("FINAL TEST SET EVALUATION")
    print("="*70 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # ========================================================================
    # Load dataset and tokenizer (EXACT SAME as training)
    # ========================================================================
    dataset = load_wikitext_safe()
    test_data = dataset["test"]
    
    combined_texts = dataset["train"]["text"]
    tokenizer = build_bpe_tokenizer(combined_texts, vocab_size=32000)
    
    PAD_ID = tokenizer.word_to_idx["<pad>"]
    
    print(f"🔤 Tokenizer:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  PAD_ID: {PAD_ID}\n")
    
    # ========================================================================
    # Create test dataset and loader (NO SHUFFLING - crucial for reproducibility)
    # ========================================================================
    print("Creating test loader...")
    test_dataset = PackedWikiTextDataset(test_data, tokenizer)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # CRITICAL: NOT shuffle test set
        num_workers=2,
        pin_memory=True
    )
    
    print(f"  Test set size: {len(test_dataset):,} sequences")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Total batches: {len(test_loader)}\n")
    
    # ========================================================================
    # Rebuild model (EXACT SAME architecture as training)
    # ========================================================================
    print("Building model...")
    model = Transformer(
        config["d_model"],
        config["num_heads"],
        config["vocab_size"],
        config["d_ff"],
        config["num_kv_heads"],
        config["num_decoder_layers"]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}\n")
    
    # ========================================================================
    # Load checkpoint
    # ========================================================================
    print("Loading checkpoint...")
    checkpoint = load_checkpoint_safely("checkpoint_latest.pt")
    
    if checkpoint is None:
        print("❌ FAILED: Could not load checkpoint!")
        return None
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"✅ Weights loaded\n")
    
    # ========================================================================
    # Evaluate
    # ========================================================================
    print("Running evaluation on test set...")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, PAD_ID)
    test_perplexity = torch.exp(torch.tensor(test_loss))
    
    # ========================================================================
    # Results
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Test Accuracy:   {test_acc:.2%}")
    print(f"Test Perplexity: {test_perplexity:.4f}")
    print("="*70 + "\n")
    
    # ========================================================================
    # Save results to file
    # ========================================================================
    test_line = (
        f"FINAL TEST | Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%} | Perplexity: {test_perplexity:.4f}\n"
    )
    
    local_results = CHECKPOINT_DIR / "epoch_results.txt"
    with open(local_results, "a") as f:
        f.write("\n" + test_line)
    
    print(f"✅ Results saved to: {local_results}\n")
    
    
    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "perplexity": float(test_perplexity)
    }


if __name__ == "__main__":
    results = eval_final()
    
    if results:
        print("✅ Evaluation complete!")
    else:
        print("❌ Evaluation failed!")