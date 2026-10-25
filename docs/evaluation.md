# How the evaluations work

- **Perplexity and benchmarks (`eval.py`).** Perplexity on a held-out FineWeb-Edu split; HellaSwag and ARC-Easy by zero-shot log-likelihood ranking of the answer choices, with both raw and length-normalised scores (the conventional one is reported for each).
- **Routing (`build_routing_eval.py`, `eval_routing.py`).** Prompts come from sources the model was not trained on (TriviaQA validation, Natural Questions, Dolly, programmatically generated arithmetic, hand-written prompts). They are deduplicated, checked for near-duplicates against every training file, and split into dev and test. The script reproduces the production first pass exactly (same decoding, same tag parser).
- **RAG (`prefetch.py`, `eval_trivia.py`).** Search results are cached once; the eval then runs the real production code path offline, comparing closed-book, always-retrieve, and routed systems, and reports where each point is lost (retrieval vs reader vs filter).
- **Statistics.** Intervals use the normal approximation for rates and a paired bootstrap for model-vs-model differences.
