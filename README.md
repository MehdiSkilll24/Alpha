# Alpha: a 317M-parameter language model, built end to end, with tool use

Alpha is a decoder-only Transformer written from scratch in PyTorch: tokenizer, pretraining on ~12B tokens of FineWeb-Edu, instruction tuning, and a small tool-using assistant on top (web search for facts, a calculator for arithmetic). Every stage has an evaluation, and this README reports the numbers, including the ones that don't flatter the model.

## Highlights

- **A strong base for its size.** Trained from scratch (317M parameters, 92k steps, ~12B tokens): perplexity **16.2** on held-out FineWeb-Edu, **38.4%** on HellaSwag and **58.3%** on ARC-Easy, zero-shot.
- **Tool routing works.** The fine-tuned model decides on its own, in its first generated tokens, whether to emit `[SEARCH]`, `[CALC]`, or answer directly. On a 450-prompt held-out routing set, routing accuracy went from **73.3% (v4) to 88.9% (v5)** on the same prompts (+15.6 points, 95% CI +12.0 to +19.3), at the cost of about one perplexity point.
- **The calculator is never invoked on non-math prompts.** Calc precision is **100%** for both versions, and calc recall rose from 50.7% to 80.7% (routing level; see Limitations for what the calculator can evaluate).
- **Search is what makes it useful.** On 968 TriviaQA questions, the base model answering from memory gets **11.9%**; the retrieval-augmented system gets **50.4%** with a search on every question (**38.5%** through the full routed system, which also refuses unsupported answers).
- **Evaluation you can trust and rerun.** Held-out splits, near-duplicate checks against the training data, paired confidence intervals, frozen search caches, and resumable runs. Everything reproduces offline once the caches are built.

## What's in the box

| Stage | What it is |
|---|---|
| Architecture | Decoder-only, 316.7M parameters: 16 layers, d_model 1280, grouped-query attention (8 query heads, 2 KV heads, head dim 160), RoPE (2048 context), pre-LayerNorm, SwiGLU MLP (d_ff 4096, hidden size 2752), untied output head, 32,000-token vocabulary, no dropout |
| Tokenizer | Byte-level BPE (Hugging Face `tokenizers`) with `<pad> <unk> <bos> <eos>` |
| Pretraining | FineWeb-Edu `sample-10BT`, 92,000 steps x 131,072 tokens/step (~12B tokens), AdamW (0.9, 0.95) with weight decay 0.1 on weight matrices, cosine LR 3e-4 to 3e-5, bf16, gradient clipping 1.0, one Colab A100. Final validation loss 2.78 (perplexity 16.2) |
| Instruction tuning | Full fine-tune, 3 epochs, LR 3e-5, loss on answer tokens only. Mix: filtered Alpaca-cleaned, SmolTalk chat, 15k TriviaQA `[SEARCH]` examples, 8k generated `[CALC]` examples, passage-grounded reader data (TriviaQA, MS MARCO, ASQA, Wikipedia), identity data, and (v5) a routing-fix set |
| Inference | KV cache, multi-turn history truncated to the 2048 context, persona system prompt, top-k / top-p / repetition-penalty sampling. Search goes through `ddgs` (Bing/Brave/Yahoo backends) and uses the first result's snippet, with a repair step for glued words. The calculator evaluates one binary operation (`a op b`) |

### How a message flows through the assistant

```mermaid
flowchart LR
    U[User message] --> R{"First pass: greedy, up to 48 tokens"}
    R -->|"[SEARCH]"| S["Web search, first result snippet"]
    S --> D["Reader: answer from the snippet"]
    D --> G{"Grounding check"}
    G -->|pass| A1[Answer]
    G -->|fail| N["I couldn't find that."]
    R -->|"[CALC] a op b"| C[Calculator] --> A2[Result]
    R -->|plain text| T["Sampled answer with chat history and persona"]
```

The router is not a separate classifier: it is the language model itself, taught by fine-tuning to start its answer with a tool tag when one is needed. The grounding check is a lexical filter: at least 60% of the answer's words, and every number in it, must appear in the retrieved text.

## Results

### 1. Base model (pretrained only)

| Metric | Result |
|---|---|
| Perplexity, held-out FineWeb-Edu | 16.19 |
| HellaSwag (zero-shot, acc_norm, n=10,042) | 38.4% |
| ARC-Easy (zero-shot, acc, n=2,376) | 58.3% |

The ARC-Easy score is probably helped by FineWeb-Edu's educational filtering.

### 2. What fine-tuning costs in general ability

| Model | Perplexity | HellaSwag (acc_norm) | ARC-Easy (acc) |
|---|---|---|---|
| Base | 16.19 | 38.4% | 58.3% |
| SFT v4 | 26.05 | 37.4% | 52.4% |
| SFT v5 (routing fix) | 27.03 | 37.3% | 50.9% |

Most of the cost comes from instruction tuning itself (base to v4). The v5 data fix added about one perplexity point, left HellaSwag unchanged, and moved ARC-Easy by about 1.5 points, which is borderline given a ~1% standard error. HellaSwag/ARC measure raw-text likelihood, not what instruction tuning is for, so judge the model by sections 3 and 5.

### 3. Routing: v4 to v5

**Setup.** A 900-prompt eval with 300 prompts per class (search, calc, answer), split into dev and test (450 each). v4 and v5 are scored on the **same 450 test prompts**.

**What changed in v5.** The same model, retrained from the base checkpoint with identical hyperparameters, with ~5k extra examples: (a) generated calc phrasings that v4 had never seen (unit conversions, word problems, question-style arithmetic such as "What does 12 * 7 equal?"), keeping only expressions the real calculator evaluates correctly; (b) wh-questions that should *not* search, converted from the existing passage data into the shape production uses (question and passages in one message); (c) non-wh phrasings of search questions and real-time queries (weather, prices, news).

| | v4 | v5 |
|---|---|---|
| **Overall accuracy** | 73.3% | **88.9%** |
| Search recall / precision | 94.7% / 70.3% | 96.0% / 88.3% |
| Calc recall / precision | 50.7% / 100% | 80.7% / 100% |
| Answer recall / precision | 74.7% / 65.1% | 90.0% / 81.3% |

v5 fixed 77 prompts v4 got wrong and newly broke 7. On a separate dev set v5 scores 89.6%, consistent with test.

| Class | Category | v4 | v5 |
|---|---|---|---|
| answer | information extraction | 7/20 | 19/20 |
| answer | closed-book QA with passage | 12/19 | 19/19 |
| answer | summarization | 12/19 | 19/19 |
| answer | classification / option lists | 13/20 | 11/20 |
| answer | brainstorming | 25/26 | 24/26 |
| answer | creative writing | 23/26 | 23/26 |
| answer | identity | 10/10 | 10/10 |
| answer | chat | 10/10 | 10/10 |
| calc | word problems | 0/21 | 17/21 |
| calc | unit conversion | 0/12 | 9/12 |
| calc | percent / tips / discounts | 6/18 | 15/18 |
| calc | powers and roots | 0/12 | 5/12 |
| calc | "What is ...?" arithmetic | 17/21 | 21/21 |
| calc | named operations ("sum of", "quotient of") | 8/15 | 10/15 |
| calc | verb leads ("Evaluate", "Solve") | 14/18 | 13/18 |
| calc | multi-step expressions | 13/15 | 13/15 |
| calc | symbolic ("17 * 23") | 18/18 | 18/18 |
| search | TriviaQA-style | 68/68 | 68/68 |
| search | Natural Questions-style | 62/67 | 61/67 |
| search | real-time (weather, news, prices) | 12/15 | 15/15 |

The largest gains are exactly where v4 sent everything to web search: questions with a pasted passage, and arithmetic phrased as a question.

*Routing only:* these rows test whether the model picked the right tool. The calculator evaluates a single binary operation (`a op b`), so for percent, power, root, and multi-step prompts a correct `[CALC]` route still ends in "Couldn't parse that expression." That is safer than the made-up numbers v4 produced for the same prompts, but it is not a correct answer.

### 4. What routing adds over a plain language model

A model without tools can only answer directly. On the balanced routing set, "always answer" would be right on the answer third only (33.3%; this is a property of the set, not a measurement of the base checkpoint). The measured evidence of what tools add:

- **Facts (measured):** on 968 TriviaQA validation questions, the base model answering from memory scores 11.9% versus 50.4% with retrieval (section 5).
- **Arithmetic (examples from v4 outputs, before the fix):** when math is not routed, a model this small computes it itself and gets it wrong:

| Prompt | Model answered directly | Correct |
|---|---|---|
| A box holds 40 tickets. How many tickets are in 5 boxes? | "There are 20 tickets in 5 boxes." | 200 |
| Calculate 25 percent of 20. | "25% of 20 is 25." | 5 |
| Find the quotient of 75 and 25. | "The quotient of 75 and 25 is 25." | 3 |

When a prompt is routed to `[CALC]` with a single operation, the calculator's result is exact, and calc precision (never invoked on non-math prompts) is 100%. Whether the model writes the *correct* expression was not evaluated.

### 5. End to end: retrieval-augmented QA on TriviaQA (v5, n=968)

Questions come from TriviaQA validation (shuffle seed 0), with web results prefetched once and frozen so the eval is reproducible. The metric is *contains* (an alias appears in the answer as whole tokens); EM and F1 are near zero by construction because the reader answers in full sentences.

| System | contains | 95% CI |
|---|---|---|
| Closed-book (base model, 5-shot) | 11.9% | +/- 2.0% |
| Always retrieve (SFT v5 reader) | 50.4% | +/- 3.1% |
| Routed (full system, with grounding check) | 38.5% | +/- 3.1% |

Where the points go:

| Diagnostic | Value |
|---|---|
| Retrieved snippet contains the answer | 69.3% |
| Reader accuracy when the answer is in the snippet | 70.2% |
| Reader accuracy when it is not | 5.7% |
| Routed system: chose to search | 99.4% |
| Routed system: refused ("I couldn't find that.") | 36.5% |
| Routed system: accuracy on the questions it answered | 60.7% |

The grounding check is a precision/coverage trade: it refuses about half of the wrong answers (240 of 480) but also 23% of the correct ones (113 of 488). It raises accuracy-when-answering from 50.4% to 60.7%, at the cost of answering only 63.5% of questions.

## How the evaluations work

- **Perplexity and benchmarks (`eval.py`).** Perplexity on a held-out FineWeb-Edu split; HellaSwag and ARC-Easy by zero-shot log-likelihood ranking of the answer choices, with both raw and length-normalised scores (the conventional one is reported for each).
- **Routing (`build_routing_eval.py`, `eval_routing.py`).** Prompts come from sources the model was not trained on (TriviaQA validation, Natural Questions, Dolly, programmatically generated arithmetic, hand-written prompts). They are deduplicated, checked for near-duplicates against every training file, and split into dev and test. The script reproduces the production first pass exactly (same decoding, same tag parser).
- **RAG (`prefetch.py`, `eval_trivia_rag.py`).** Search results are cached once; the eval then runs the real production code path offline, comparing closed-book, always-retrieve, and routed systems, and reports where each point is lost (retrieval vs reader vs filter).
- **Statistics.** Intervals use the normal approximation for rates and a paired bootstrap for model-vs-model differences.

## Limitations

These are measured, not hypothetical.

**Model and reader**
- It is a 317M-parameter model. Even when the retrieved snippet contains the answer, the reader misses it 30% of the time, typically by restating the question, inventing an entity, or rambling past the answer (the answers are long: median 35 words).
- When the snippet lacks the answer (31% of questions), the reader answers anyway: 149 confidently wrong answers got past the grounding check. A lexical overlap check cannot detect an invented answer.
- Retrieval uses only the first search result's snippet.

**Routing**
- Calc recall is 80.7%, not 100%. Weakest: powers and roots (5/12), named operations (10/15), and "Evaluate/Solve" leads (13/18). Unrouted arithmetic is answered wrongly by the model itself.
- **The calculator parses a single binary operation (`a op b`) with `+ - * /`.** Percentages, powers, roots, and multi-step expressions can be routed to it correctly but not evaluated: they end in "Couldn't parse that expression." (safer than a made-up number, but not an answer). The routing eval checks the tool tag only, not whether the model wrote the right expression, so end-to-end calculator accuracy was not measured.
- Option-list classification prompts ("Identify which instrument is string or percussion: ...") still go to search (9 of the 15 answer-to-search misses; this category got slightly worse in v5).
- Search recall on lowercase, Natural Questions-style prompts is 61/67 and some of those prompts are arguably ambiguous.
- Routing ignores conversation history, so a follow-up like "and what about Germany?" has no context for the router.

**Evaluation caveats**
- **The routing test set was inspected during development**: v4's test failures guided what data v5 added. Dev (89.6%) and test (88.9%) agree, which suggests no large inflation, but treat test as "inspected", not pristine.
- **The routing eval is partly in-distribution for calc**: the generated calc phrasings in the training fix and the eval share template families, and 22 of the 450 test prompts are near-duplicates of training prompts (v5 accuracy excluding them: 88.3%). The hand-written prompts are the real out-of-distribution check.
- HellaSwag/ARC use a custom scorer, not identical to `lm-evaluation-harness`; compare with published numbers cautiously.
- TriviaQA is in-distribution for the router and reader (both were trained on TriviaQA-derived data), so routing-to-search on it is not evidence of generalisation, and *contains* is lenient with long answers: counting only the first sentence gives 44.4% for always-retrieve instead of 50.4%.
- Single training run per version; no seed variance is measured. Intervals are +/-3% at these sample sizes.

**Cost of fine-tuning**
- Instruction tuning raises raw-text perplexity from 16.2 to 27.0 and lowers ARC-Easy from 58.3% to 50.9%. Techniques that limit forgetting (data replay, LoRA) were not tried.

**Scope**
- A research and learning project: no safety or alignment work, and it will hallucinate. Context length is 2048 tokens. Live web search is not deterministic; the evals use a frozen cache.

## What I would do next

1. Extend the calculator (multi-step expressions, percentages, powers, roots), add matching training data and phrasing diversity, and evaluate the expressions the model writes end to end, on fresh hand-written prompts.
2. Use the top 2-3 snippets instead of one to raise the retrieval ceiling.
3. Tune the grounding check offline (`tune_grounding.py`: answer truncation and thresholds, chosen on one half of the questions and verified on the other) and rerun the RAG eval.
4. Compare full fine-tuning, fine-tuning with pretraining-data replay, and LoRA, judged on routing accuracy and perplexity.

## Reproducing

Layout used by the scripts:

```
Alpha/
  model/        # code (main.py, train.py, sft.py, inference.py, search.py, calculator.py, evals)
  jsons/        # tokenizer, config, SFT datasets
  checkpoints/  # checkpoint_latest.pt (base), checkpoint_sftv4.pt, checkpoint_sftv5.pt
```

```bash
pip install -r requirements.txt

# 1. data + pretraining
python fetchds.py                  # tokenize FineWeb-Edu into train.bin / val.bin
python train.py                    # resumes from checkpoint_latest.pt

# 2. SFT data, then fine-tune
python search_ex.py; python calc_ex.py; python identity.py; python marcods.py
python rag_ex.py        # needs the local retrieval index (rag_local.py + index.pt)
python asqa.py; python rag_wiki.py; python make_chat.py
python build_routing_fix.py --out ../jsons/routing_fix.json
python sft.py --out ../checkpoints/checkpoint_sftv5.pt

# 3. evaluations
python eval.py --ckpt ../checkpoints/checkpoint_sftv5.pt --out eval_sftv5.json
python build_routing_eval.py --hand routing_prompts.json --train-files ../jsons/*.json
python eval_routing.py --ckpt ../checkpoints/checkpoint_sftv5.pt --prompts routing_test.json
python prefetch.py
python eval_trivia_rag.py --sft ../checkpoints/checkpoint_sftv5.pt --base ../checkpoints/checkpoint_latest.pt

# 4. chat
python inference.py --ckpt ../checkpoints/checkpoint_sftv5.pt
```

Result files (`eval_*.json`, `routing_*.json`, `trivia_rag_results.json`) and the search cache (`hits_snippets.json`) are committed so the numbers above can be checked without rerunning anything. Model weights (base and fine-tuned checkpoints) are on the Hugging Face Hub: [MSkilll/Etra](https://huggingface.co/MSkilll/Etra/tree/main/checkpoints).

## License and attribution

| Component | License | What it means |
|---|---|---|
| Code (this repository) | [MIT](LICENSE) | Free to use, modify, and distribute. Keep the copyright notice in copies. |
| Model weights | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | Attribution required; non-commercial use only. The fine-tuning data includes Alpaca-derived data, which is non-commercial and restricts models trained on it to research use. |

**If you use or build on this work, please credit it as the original**: name the project (Alpha by MSkilll), link to this repository and the [weights](https://huggingface.co/MSkilll/Etra), and state any changes you made. MIT already requires the copyright notice to stay in copies of the code; the weights license requires attribution; and for papers, reports, or derived projects, please cite:

```bibtex
@misc{alpha2026,
  title        = {Alpha: a 317M-parameter language model with tool use, built end to end},
  author       = {MSkilll},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/MSkilll/Etra}}
}
```

### Data sources

Licenses as listed on each source's page at the time of writing; check them before relying on this table.

| Data | Used for | License / terms |
|---|---|---|
| FineWeb-Edu (`sample-10BT`) | Pretraining | ODC-By 1.0 (plus Common Crawl terms of use) |
| Alpaca-cleaned | Instruction tuning | CC BY-NC 4.0, following the original Stanford Alpaca data (generated with OpenAI's text-davinci-003, so OpenAI's terms may also apply) |
| SmolTalk | Chat tuning | Apache 2.0 |
| TriviaQA | Search and reader data, RAG eval | No license stated on its Hugging Face card; research use |
| MS MARCO | Reader data | Microsoft's terms (non-commercial research) |
| ASQA | Reader data | See the dataset card |
| Wikipedia | Reader data | CC BY-SA |
| HellaSwag | Evaluation only | MIT |
| ARC | Evaluation only | CC BY-SA 4.0 |
| Natural Questions (open), Dolly | Evaluation only | CC BY-SA 3.0 |

I am not a lawyer; this section describes intent and sources, not legal advice.
