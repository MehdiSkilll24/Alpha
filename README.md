# Alpha Transformer

A custom **Encoder-Decoder Transformer model trained from scratch** for English → French translation.

Alpha is a 51M parameter Transformer implemented entirely in PyTorch, including:
- Multi-Head Attention
- Rotary Positional Embeddings (RoPE)
- Encoder-Decoder architecture
- Cross Attention
- Custom tokenizer
- Mixed precision training
- Checkpointing system

The model was trained on the **OPUS-100 English-French translation dataset**.

---

## Model Overview

| Feature | Value |
|---|---|
| Architecture | Encoder-Decoder Transformer |
| Parameters | 51.26M |
| Framework | PyTorch |
| Task | English → French Translation |
| Dataset | OPUS-100 |
| Training Samples | ~1M sentences |
| Embedding Dimension | 256 |
| Attention Heads | 8 |
| Feed Forward Dimension | 1024 |
| Encoder Layers | 3 |
| Decoder Layers | 3 |
| Maximum Sequence Length | 64 tokens |
| Positional Encoding | RoPE |

---

# Performance

Evaluation on the OPUS-100 test set:

| Metric | Result |
|---|---|
| Token Accuracy | 62.8% |

The model was trained from scratch without using pretrained weights.

---

# Architecture

## Encoder

Each encoder block contains:

Input Embeddings
|
v
LayerNorm
|
Multi-Head Self Attention + RoPE
|
Residual Connection
|
LayerNorm
|
Feed Forward Network
|
Residual Connection

## Decoder

Each decoder block contains:

Target Embeddings
|
LayerNorm
|
Masked Self Attention + RoPE
|
Residual Connection
|
LayerNorm
|
Cross Attention
|
Residual Connection
|
LayerNorm
|
Feed Forward Network
|
Residual Connection

---

# Files


Alpha/
│
├── config.json
│ └── Model architecture configuration
│
├── model.safetensors
│ └── Trained model weights
│
├── src_vocab.json
│ └── English vocabulary
│
├── tgt_vocab.json
│ └── French vocabulary
│
├── tokenizer.py
│ └── Custom tokenizer implementation
│
├── main.py
│ └── Transformer architecture
│
├── inference.py
│ └── Translation script
│
└── README.md

---

# Installation

Clone the repository:

```bash
git clone https://github.com/MehdiSkilll24/Alpha.git
cd Alpha

Install dependencies:

pip install torch safetensors

Usage

Run inference:

python inference.py

Example:

Input:
> hello, how are you?

Output:
> bonjour, comment allez-vous ?



Loading the Model Manually
import torch
import json
from safetensors.torch import load_file
from main import Transformer

with open("config.json") as f:
    config = json.load(f)

weights = load_file("model.safetensors")

model = Transformer(
    src_vocab_size=config["src_vocab_size"],
    tgt_vocab_size=config["tgt_vocab_size"],
    d_model=config["d_model"],
    num_heads=config["num_heads"],
    d_ff=config["d_ff"]
)

model.load_state_dict(weights)
model.eval()


Training

The model was trained using:

AdamW optimizer
Mixed precision training (FP16)
ReduceLROnPlateau scheduler
Custom PyTorch training loop

Training configuration:

Learning Rate: 3e-4
Batch Size: 32
Epochs: 18
Dataset

Training data:

OPUS-100 English-French translation dataset.

The dataset contains parallel English/French sentence pairs collected from various sources.

Limitations

This model is designed as an educational and experimental Transformer implementation.

Limitations:

Vocabulary is based on a custom word-level tokenizer
Limited context length
No pretrained initialization
Smaller dataset compared to production translation systems
Token accuracy does not directly represent BLEU score
Future Improvements

Possible improvements:

Replace word tokenizer with BPE/SentencePiece
Increase model depth
Train on larger datasets
Add beam search decoding
Evaluate with BLEU / COMET
Implement KV-cache for faster inference
Motivation

Alpha was built to understand and implement the Transformer architecture from first principles.

Rather than using existing libraries, the project implements the core components manually:

Attention calculations
Positional embeddings
Encoder/decoder blocks
Training pipeline
Model serialization

The goal is to better understand how modern language models are built internally.

Source Code

GitHub:

https://github.com/MehdiSkilll24/Alpha

Model weights:

https://huggingface.co/MehdiSkilll24/Alpha

License

MIT License


One suggestion before you upload: add **3-5 real translation examples** from your model. That will make the HF page look 10x more alive. A model card with "62.8% accuracy" is okay; a model card with:


English:
"The weather is beautiful today."

Alpha:
"Le temps est magnifique aujourd'hui."