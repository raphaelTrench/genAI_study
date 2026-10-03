# RAG study

This folder is a study project for implementing retrieval-augmented generation
(RAG) approaches and comparing their effectiveness. The existing
[`naive_rag.ipynb`](./naive_rag.ipynb) is the baseline experiment.

## Structure

```text
chat-with-your-docs/
├── naive_rag.ipynb
├── project.md
└── evaluation/
    ├── evaluate.py
    └── test_evaluate.py
```

Keep one notebook per RAG strategy (for example, hybrid retrieval,
multi-query retrieval, or reranking). Shared, deterministic evaluation helpers
belong in `evaluation/`; keep corpus files under the repository's `data/`
directory and generated reports under an ignored or otherwise untracked
`results/` directory.

The project's default text-generation and embedding models are configured in
the repository-root [`config.py`](../config.py) as
`DEFAULT_TEXT_GENERATION_MODEL_ID` and `DEFAULT_EMBEDDING_MODEL_ID`. Import
these constants in new project code rather than duplicating model IDs.

The same configuration defines the source text directory, generated chunk JSON
path, chunk size, and overlap. Run `make chunks` to regenerate the chunk file;
the notebook reads that file instead of loading and splitting source text
itself.

## Comparison protocol

For each experiment, record the strategy, embedding model, generation model,
chunking configuration, retrieval parameters, and evaluation date. Compare
strategies on the same corpus and question set. When comparing retrieval or
generation techniques, keep chunking, models, context budget, and other
settings fixed unless that is the variable under study.

Maintain a manually reviewed benchmark with each question's relevant chunk
IDs. Use the same chunking configuration and stable chunk IDs across retrieval
strategy comparisons. Chunk IDs are not comparable when chunking changes;
chunking experiments need evidence-span labels that can be mapped to each
configuration's chunks.

Report retrieval and generation quality separately:

- **Retrieval:** `Recall@k`, `Precision@k`, and `MRR@k`, computed with `ranx`
  by the `Evaluator` class in `evaluation/evaluate.py`. It loads the dataset
  from the configured `GROUND_TRUTH_JSON_PATH`; each experiment notebook can
  create `Evaluator(k_values=(1, 3, 5))` and call `.evaluate(retriever)`.
  Queries run concurrently with four workers by default; set `max_workers` to
  match the concurrency supported by the retriever.
- **Generated answers:** assess faithfulness/groundedness, answer relevance,
  and correctness against a reference answer. These may use human review or an
  LLM judge; record the judge model and prompt because judge scores are not
  model-independent.
- **Operational tradeoffs:** record latency and, where available, token usage
  and cost. Do not collapse all results into one score without explaining the
  weighting.

The retrieval metric helper is provider-independent and makes no paid API
calls. Run its tests with:

```bash
poetry run python -m unittest discover -s evaluation -p 'test_*.py'
```
