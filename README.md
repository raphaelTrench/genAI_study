# genAI_study

## Synthetic query generation

Generate the notebook's chunk input from the source text files:

```bash
make chunks
```

This scans `data/mistborn/` recursively and writes chunks to
`evaluation/generation/chunks.json` with stable IDs and source filenames. The
generated file is ignored by Git. The source directory, output path, chunk
size, and overlap are configured in the root `config.py`; override them for a
single run with `CHUNK_SOURCE`, `CHUNKS_OUTPUT`, `CHUNK_SIZE`, and
`CHUNK_OVERLAP`.

The chunk generator writes a list of `{doc_id, source, text}` objects. For a
custom chunk dataset, the query generator also accepts a JSON list or an object
with a `chunks` list:

```json
[
  {
    "doc_id": "doc_101",
    "text": "Task decomposition involves breaking complex tasks into smaller subgoals."
  }
]
```

Generate queries and their relevance judgments from the generated chunks with:

```bash
make generate
```

The default output is `results/synthetic_ground_truth.json`. To customize the
model, device, or output path, set `MODEL_ID`, `DEVICE` (`-1` for CPU, `0` for
the first GPU), or `OUTPUT`:

```bash
make generate INPUT=data/chunks.json OUTPUT=results/queries.json DEVICE=0 \
  BATCH_SIZE=16 ARGS=--local-files-only
```

The project's default text-generation model is
`Qwen/Qwen2.5-1.5B-Instruct`, and its default embedding model is
`sentence-transformers/all-MiniLM-L6-v2`. Both are defined in the repository
root's `config.py` as `DEFAULT_TEXT_GENERATION_MODEL_ID` and
`DEFAULT_EMBEDDING_MODEL_ID`, so other project code can import these shared
defaults rather than duplicating model IDs. The generation model downloads on
first use unless `--local-files-only` is provided through `ARGS`.
Generation defaults to GPU 0 with batches of 8 prompts and at most 128 new
tokens per chunk; these settings are configured in `config.py`. Increase
`BATCH_SIZE` gradually if GPU memory allows. Set `DEVICE=-1` to force CPU
generation, or `DEVICE=1` to select the second GPU.

For retrieval evaluation, generate the ground-truth dataset with `make generate`,
then use `Evaluator(k_values=(1, 3, 5)).evaluate(retriever)` from
`evaluation/evaluate.py`. The evaluator uses `ranx` to calculate
Precision@k, Recall@k, and MRR@k, and loads the dataset from the configured
`GROUND_TRUTH_JSON_PATH`; pass `ground_truth_path=...` to load another dataset.
It retrieves queries concurrently with four workers by default; adjust this
using `Evaluator(max_workers=2)` if the retriever has a lower concurrency limit.
The RAG notebook's final cell demonstrates this shared workflow.