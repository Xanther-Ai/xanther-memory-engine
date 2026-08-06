# LongMemEval Benchmark for XME

Reproducible harness for running [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
(ICLR 2025) against Xanther Memory Engine.

## What LongMemEval tests

500 questions across 5 memory abilities:

| Category | What it tests |
|----------|--------------|
| `single-session-user` | Recall a fact from a specific session |
| `multi-session-user` | Combine facts across multiple sessions |
| `single-session-assistant` | Remember what the assistant said |
| `temporal-reasoning` | Reason about when something happened |
| `knowledge-updates` | Handle contradictory/updated information |

Each question has ~54 haystack sessions (~115K tokens total) to search through.

## Quick start (10 questions, ~5 min)

```bash
cd /path/to/xanther-memory-engine
pip install -e ".[llm]"

python -m benchmarks.longmemeval.run \
    --data benchmarks/longmemeval/data/raw/longmemeval_s \
    --api-key $OPENROUTER_API_KEY \
    --limit 10 \
    --output-dir benchmarks/longmemeval/results/smoke_test
```

## Full run (500 questions, ~2-3 hours)

```bash
python -m benchmarks.longmemeval.run \
    --data benchmarks/longmemeval/data/raw/longmemeval_s \
    --api-key $OPENROUTER_API_KEY \
    --output-dir benchmarks/longmemeval/results/run_001 \
    --answer-model openai/gpt-4o-mini \
    --judge-mode string
```

For LLM judge scoring (more accurate, higher cost):
```bash
    --judge-mode llm \
    --judge-model openai/gpt-4o-mini
```

## Run steps individually

```bash
# Step 1: Ingest all sessions into XME
python -m benchmarks.longmemeval.ingest \
    --data benchmarks/longmemeval/data/raw/longmemeval_s \
    --project-id longmemeval-xme \
    --limit 50

# Step 2: Retrieve context + generate answers
python -m benchmarks.longmemeval.retrieve \
    --data benchmarks/longmemeval/data/raw/longmemeval_s \
    --project-id longmemeval-xme \
    --api-key $OPENROUTER_API_KEY \
    --output benchmarks/longmemeval/results/predictions.jsonl \
    --limit 50

# Step 3: Score predictions
python -m benchmarks.longmemeval.evaluate \
    --predictions benchmarks/longmemeval/results/predictions.jsonl \
    --mode string    # or --mode llm --api-key $OPENROUTER_API_KEY
```

## Resuming interrupted runs

All three phases support `--resume` (default on). Re-run the same command and
it picks up where it left off. Ingestion tracks progress in
`.xanther/bench_<project_id>.db` and `data/ingest_progress_*.json`.

## Cost estimate

| Component | Cost (500 questions) |
|-----------|---------------------|
| Answer generation (gpt-4o-mini, ~200 tokens/answer) | ~$0.15 |
| LLM judge (gpt-4o-mini, ~100 tokens/judgment) | ~$0.08 |
| **Total** | **~$0.23** |

String match mode costs $0 for evaluation (judge step free).

## Published scores for comparison

| System | LongMemEval accuracy | Judge | Notes |
|--------|---------------------|-------|-------|
| Zep (Graphiti) | 63.8% | LLM | Temporal graph, $25/mo |
| Mem0 (2025) | 49.0% | LLM | Vector-first |
| Mem0 (2026*) | 93.4% | LLM | Changed methodology, not reproducible |
| Letta | 83.2% | LLM | Full agent runtime |
| **XME v0.1** | **TBD** | string | Run the harness to find out |

*Mem0's 2026 number used a modified "LongMemEval-S*" variant not available to others.
This harness uses the original ICLR 2025 dataset.

## Methodology

- Dataset: `longmemeval_s` (500 questions, standard variant)
- Ingestion: all haystack sessions via `xme_session_end`
- Retrieval: `xme_search(question, layers=["episodic","facts"], top_k=10)`
- Answer generation: fixed system prompt, temperature=0
- Scoring: substring/normalized string match (or LLM judge)

All parameters logged in `results/*/summary.json` for full reproducibility.
