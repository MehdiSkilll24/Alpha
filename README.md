# Alpha Transformer

A **51M-parameter Encoder-Decoder Transformer trained from scratch** on English→French translation. Built entirely in PyTorch with custom attention, positional encodings, and training pipeline—no pretrained weights, no abstraction libraries.

**GitHub:** [MehdiSkilll24/Alpha](https://github.com/MehdiSkilll24/Alpha)  
**Model Weights:** [MSkilll/Alpha55](https://huggingface.co/MSkilll/Alpha55)

---

## Translation Examples

| English Input | Model Output |
|---|---|
| "hi" | bonjour |
| "how are you doing on this fine day?" | comment allez-vous ce beau jour? |
| "I am taking a shower while listening to music" | je prends une douche en écoutant la musique |
| "This is indeed a fine day" | c'est effectivement un bon jour |
| "I would definitely go if I could, but you know how busy I am these days" | je m'en certainement si je pouvais, mais vous savez combien je suis occupé ces jours |

---

## Architecture

**Type:** Encoder-Decoder Transformer with Cross-Attention  
**Parameters:** 51.26M  
**Framework:** PyTorch (from-scratch implementation)  
**Positional Encoding:** Rotary Positional Embeddings (RoPE)  
**Training:** Mixed Precision (FP16) + AdamW + ReduceLROnPlateau  

| Component | Configuration |
|---|---|
| Embedding Dimension | 256 |
| Attention Heads | 8 |
| Feed Forward Dimension | 1024 |
| Encoder Layers | 3 |
| Decoder Layers | 3 |
| Max Sequence Length | 64 tokens |
| Vocabulary | Word-level custom tokenizer |

### Encoder Block
```
Input → LayerNorm → Multi-Head Self-Attention (RoPE) → Residual → 
LayerNorm → FFN → Residual → Output
```

### Decoder Block
```
Target → LayerNorm → Masked Self-Attention (RoPE) → Residual →
LayerNorm → Cross-Attention (encoder output) → Residual →
LayerNorm → FFN → Residual → Output
```

---

## Performance

| Metric | Value |
|---|---|
| **Token Accuracy** | 62.8% |
| **Dataset** | OPUS-100 (English-French) |
| **Training Samples** | ~1M parallel sentences |
| **Test Set Accuracy** | Evaluated on OPUS-100 test split |

### About Token Accuracy
Token accuracy (62.8%) measures exact-match correctness at the token level on a word-level tokenizer with 64-token sequences. This differs from BLEU scores used in production translation systems. The examples above demonstrate that the model captures semantic and syntactic patterns despite token-level limitations—particularly on shorter, high-frequency phrases.

---

## Installation

```bash
git clone https://github.com/MehdiSkilll24/Alpha.git
cd Alpha
pip install torch safetensors
```

### Dependencies
- `torch>=2.0`
- `safetensors`

---

## Usage

### Inference via Python
```python
import torch
import json
from safetensors.torch import load_file
from main import Transformer

# Load config and weights
with open("config.json") as f:
    config = json.load(f)

weights = load_file("model.safetensors")

# Initialize model
model = Transformer(
    src_vocab_size=config["src_vocab_size"],
    tgt_vocab_size=config["tgt_vocab_size"],
    d_model=config["d_model"],
    num_heads=config["num_heads"],
    d_ff=config["d_ff"],
    num_encoder_layers=config["num_encoder_layers"],
    num_decoder_layers=config["num_decoder_layers"]
)

model.load_state_dict(weights)
model.eval()

# Generate translation
with torch.no_grad():
    output = model.generate(input_ids, max_length=64)
```

### Command-Line Inference
```bash
python inference.py
> Enter English text: "hello, how are you?"
> Translation: bonjour, comment allez-vous?
```

---

## Training Details

**Dataset:** OPUS-100 English-French parallel corpus  
**Optimizer:** AdamW (lr=3e-4)  
**Batch Size:** 32  
**Epochs:** 18  
**Mixed Precision:** FP16  
**Scheduler:** ReduceLROnPlateau (patience=2, factor=0.5)  

The model was trained from scratch without pretrained initialization—all parameters learned from random initialization on the translation task.

---

## What Makes This Project Noteworthy

1. **Full-Stack Implementation:** Attention mechanics, positional embeddings, layer normalization, and training loops written from first principles—not abstracted away by high-level APIs.

2. **No Pretrained Crutch:** Unlike fine-tuning projects, this demonstrates understanding of how transformers actually work by implementing core components manually.

3. **Production Patterns:** Mixed precision training, checkpointing, and model serialization (safetensors) follow industry standards.

4. **Architectural Clarity:** Encoder-decoder with cross-attention explicitly separates source encoding from target generation—ideal for understanding seq2seq flows.

---

## Limitations & Design Tradeoffs

- **Word-Level Tokenizer:** Limits vocabulary and generalization. Production systems use BPE/SentencePiece.
- **Short Context:** 64-token max length constrains longer sentences; practically fine for this dataset distribution.
- **Small Scale:** 51M parameters and 1M training samples vs. modern systems (billions of parameters, trillions of tokens).
- **Token Accuracy Ceiling:** Word-level tokenizer and sequence length create a practical accuracy upper bound; see examples for actual translation quality.

---

## Project Structure

```
Alpha/
├── main.py              # Transformer architecture implementation
├── tokenizer.py         # Custom word-level tokenizer
├── inference.py         # Translation inference script
├── config.json          # Model hyperparameters
├── model.safetensors    # Trained weights
├── src_vocab.json       # English vocabulary
├── tgt_vocab.json       # French vocabulary
└── README.md            # This file
```

---

## Future Work & Architectural Ablations

**Planned branches:**
- **`decoder-only`:** Causal-masked decoder operating on concatenated source+target sequences; comparison against cross-attention efficiency
- **`encoder-only`:** Encoder-based representation model; reformulated task to extract semantic embeddings
- **Ablation Study:** Controlled parameter matching and systematic comparison on identical data splits

Each variant explores how architectural choice affects translation performance and computational efficiency—valuable for understanding when to choose which design.

---

## Motivation

Alpha was built to master the Transformer architecture from first principles rather than relying on library abstractions. Too many practitioners use `transformers.AutoModel.from_pretrained()` without understanding what happens inside.

By implementing attention, residual connections, layer normalization, positional encodings, and the full training loop manually, this project forces engagement with the real mechanics of modern language models.

---

## License

MIT License

---

## Citation

If you use this model or code in your work:

```bibtex
@misc{alpha-transformer-2024,
  author = {Mehdi Skilll},
  title = {Alpha: A From-Scratch Encoder-Decoder Transformer for English-French Translation},
  year = {2024},
  howpublished = {\url{https://github.com/MehdiSkilll24/Alpha}},
}
```
