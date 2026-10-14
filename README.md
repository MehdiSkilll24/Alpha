# Alpha: a small decoder-only LLM with tool use (web search + calculator)

A decoder-only transformer written in PyTorch, pretrained on FineWeb-Edu, then
instruction-tuned to **decide for itself when to call a tool**. It emits
`[SEARCH] <query>` for factual questions and `[CALC] a op b` for arithmetic. A thin Python
router executes the call, and the model then reads the retrieved passages and answers.

The point of the project is the full stack, measured honestly: architecture, pretraining,
SFT data construction, a tool-use loop, and an evaluation with ablations. It is not meant
to compete with large models.

## Results

Evaluation: TriviaQA validation, 1,000 questions (shuffle seed 0), **950 graded**: the
ones for which web retrieval succeeded. Web results were fetched once and frozen
(`hits_cache.json`), so every system sees identical passages. Metric `contains` = any gold
alias appears in the answer. Greedy decoding. 95% interval is about ±3 points at this n.

| System | contains | EM | F1 | avg. answer length |
|---|---|---|---|---|
| Closed-book (pretrained base, 3-shot prompt) | 0.139 | 0.069 | 0.151 | 3.3 words |
| **SFT v3, always retrieve** | **0.551** | 0.423 | 0.509 | 4.9 words |
| SFT v3, routed (model decides) | 0.500 | 0.381 | 0.460 | 4.5 words |
| SFT v4, always retrieve (experimental, see notes) | 0.499 | 0.041 | 0.208 | 15.5 words |

What the numbers say:
- **Retrieval is the whole story for a model this size:** 13.9% closed-book vs 55.1% with
  passages, a ~4x gain that is far outside the noise.
- **The reader is the bottleneck, not retrieval.** The gold answer appears in the three
  retrieved passages for **92.8%** of questions, but the model answers correctly on 55%.
- **Routed vs. always-retrieve** differ by 5 points, but that gap is a cache artifact: the
  router sometimes rewrites the query, which then misses the frozen cache (92 of 950
  questions). On the 861 questions where both found passages, the gap is ~2 points (v3).
- **v4 trains the model to answer in full sentences** (and on definition questions). EM and
  F1 collapse by design, and `contains` drops 5 points. Most of the drop is the reader copying
  page titles: the cached passages still carry title prefixes that v4 never saw during
  retrieval-time use. A rerun on a title-free cache is pending, so treat v4 as unresolved.


## How it works

```
user prompt
   |
   v
 Alpha (greedy)  ->  "[CALC] 437 / 29"   -> regex-validated, evaluated in Python
   |            \->  "[SEARCH] <query>"  -> DuckDuckGo -> fetch pages -> chunk -> rank (bge-small)
   |                                          -> top 3 passages -> Alpha reads them -> answer
   \->  otherwise: answer directly
```

- **Model (`main.py`):** pre-LayerNorm decoder blocks, grouped-query attention, rotary
  position embeddings (interleaved pairing), SwiGLU MLP, scaled init on the residual
  projections. See config.json
- **Tokenizer:** BPE trained on the pretraining corpus.
- **Pretraining (`train.py`):** FineWeb-Edu, sequence length 2048, AdamW (beta 0.9/0.95,
  weight decay 0.1 on matrices only), warmup + cosine decay to 10% of peak LR 3e-4, bf16
  autocast, gradient accumulation, gradient clipping, resumable deterministic sampler.
  Configured for 92k steps at 131k tokens per step (about 12B tokens) on rented A100.
  final validation loss / perplexity 1.59 for ~ 17 ppx
- **SFT (`sft.py`):** loss on answer tokens only (prompt tokens masked with `-100`),
  examples that would exceed the context are dropped rather than truncated, LR 3e-5,
  3 epochs, effective batch 32, best-validation checkpointing.
- **Retrieval (`rag.py`):** live web search, page extraction with trafilatura, 150-word
  chunks with overlap, table-like chunks filtered out, bge-small embeddings for ranking.
- **Calculator (`calculator.py`):** the model's expression must fully match
  `number op number`; nothing is passed to `eval`.

### SFT data mix

| Source | Role |
|---|---|
| Alpaca-cleaned (filtered for factual and arithmetic prompts that would conflict) | general instruction following |
| 15k TriviaQA questions -> `[SEARCH] <question>` | teaches *when* to search |
| 8k generated arithmetic prompts -> `[CALC] a op b` (plus a held-out set with unseen phrasings) | teaches *when* to calculate |
| TriviaQA + retrieved passages -> short answer | reader, one-span answers |
| ASQA + MS MARCO -> complete-sentence answers | reader, multi-fact answers |
| Wikipedia lead sentences (v4) | definition-style answers |

The prompt format is shared between training and inference (`prompts.py`). A one-character
mismatch changes behavior, so it is defined once.

## Reproduce

```bash
pip install -r requirements.txt

python train.py                                   # pretraining (long)
python make_search_data.py && python make_calc_data.py && python make_rag_data.py
python sft.py --base checkpoints/checkpoint_latest.pt --out checkpoints/checkpoint_sft.pt
python prefetch.py                                # fetch web hits once (hours; rate limited)
python eval_qa.py --sft checkpoints/checkpoint_sft.pt --base checkpoints/checkpoint_latest.pt
python inference.py                               # interactive
```


## What failed, and why

Documented because the failures taught more than the successes.

- **Routing by phrasing, not by need.** The first router sent "What's 12 times 8" to search,
  because it matched trivia-style wording. Fix: generate arithmetic data with many surface
  forms and test on held-out phrasings.
- **One-span readers.** Every TriviaQA reader target was a single span, so "who won..."
  questions returned one name where two were correct. Fix: add multi-fact sentence data.
- **Confident answers when retrieval misses.** Training answers were always present in the
  passages, so the model invents one when they are not. A word-overlap check catches
  invented text, but it cannot catch a wrong span that is present.
- **Entity echo.** The reader sometimes copies a phrase from the question as its answer.
- **Title copying (v4).** Passages that begin with a page title get copied as the answer
  after training on "first sentence" style targets.
- **Repetition loops** in free-form answers at low temperature, a known small-model failure.
  Mitigated with sampling and a repetition penalty on the direct-answer path only (the
  penalty hurts span copying).
- **Evaluation hygiene.** Early comparisons used live web search, whose failures and rate
  limits changed the scores. Results are now computed from a frozen cache and reported
  with the number of dropped questions.

## Limitations

- The model has 316M parameters . Without retrieval it answers few trivia
  questions correctly (14%).
- Graded on questions where retrieval succeeded (950 of 1,000), which flatters retrieval
  slightly.
- The closed-book baseline is the base model with a 3-shot prompt, not an instruction-tuned
  closed-book model, so it is a weak baseline.
- Only English trivia was evaluated. Definition, how-to and multi-turn behavior are
  checked by hand, not measured.
- The web is a moving target. The frozen cache makes results reproducible, but it does not
  reflect current search results.
- Output text from web pages is placed straight into the prompt, so a malicious page could
  contain instructions. This is a toy system and has no defenses.

## Development notes

The architecture and training loop were written with AI assistance (Claude) and reviewed
line by line. The data design, SFT pipeline, tool-use loop, evaluation and debugging were
done iteratively by the author.

Data sources: FineWeb-Edu, Alpaca-cleaned, TriviaQA, ASQA, MS MARCO, Wikipedia (CC BY-SA).
Please check each license before reusing the data or weights.
