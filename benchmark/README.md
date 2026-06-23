# Benchmark: LoCoMo Evaluation

This directory contains the evaluation code that produced the results reported in:

> **Nous: A Predictive World Model for Long-Term Agent Memory**
> Pranav Singh — IIT Ropar
> [arXiv:2606.22030](https://arxiv.org/abs/2606.22030)

---

## Results (as reported in the paper)

| Category | F1 | BLEU-1 | Context Recall | n |
|----------|----|--------|----------------|---|
| Single-hop | 63.50 | 45.60 | 86.07 | 841 |
| Multi-hop | 55.32 | 23.90 | 75.35 | 282 |
| Temporal | 58.57 | 56.04 | 61.82 | 321 |
| Open-domain | 62.50 | 10.06 | 47.50 | 96 |
| **Macro avg** | **59.97** | 33.90 | 67.69 | 1540 |

Backbone: **GPT-4o-mini** for extraction, answer generation, and judging.
Evaluation: strict token-level F1 (not LLM-as-judge).

---

## Prerequisites

```bash
pip install nous-state openai tiktoken rank_bm25
```

You need an OpenAI API key (or any OpenAI-compatible endpoint):

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Get the LoCoMo dataset

The LoCoMo dataset is from [Maharana et al., 2024](https://arxiv.org/abs/2402.17753).
Download it and place it at `benchmark/locomo/`:

```bash
# Official dataset repo
git clone https://github.com/snap-research/locomo benchmark/locomo
```

The eval script expects the 10-conversation split used in the paper.

---

## Run the evaluation

```bash
# From the repo root
python -u benchmark/eval_locomo.py
```

This will:
1. Ingest all sessions for each of the 10 conversations into a fresh Nous world model
2. Run all 1,540 QA questions using the BM25 + 2-hop BFS retrieval pipeline
3. Score answers with token-level F1 and BLEU-1
4. Write results to `benchmark/results_latest.jsonl`
5. Print per-category scores to stdout

Expected runtime: approximately 3–5 hours with GPT-4o-mini (API costs roughly $2–4 USD for the full 10-conversation run).

---

## Evaluation pipeline (what the script does)

The eval differs from the installable `nous-state` package in one important way: it adds a research-quality retrieval layer on top of the core Bayesian engine:

1. **Ingestion** — LLM extracts `(entity, attribute, value)` triples from each session turn. Each triple triggers a surprise computation and Bayesian posterior update. A delta is appended to the log.

2. **Retrieval** — At query time, named entities are extracted from the question and used to seed a 2-hop BFS over the entity coupling graph. Entity profiles (fact lines) are assembled using observed-values aggregation (Eq. 7 in the paper). The top-k delta records by BM25 similarity to the question are added as evidence lines.

3. **Category routing** — The question is classified (single-hop / multi-hop / temporal / open-domain) and routed to a category-specific system prompt that instructs the answering LLM on how to extract or aggregate the answer.

4. **Scoring** — Token-level F1 and BLEU-1 are computed against ground-truth answers. A GPT-4o-mini judge assigns each answer to one of four failure buckets: good, answer miss, unknown, retrieval miss.

---

## Notes on reproducibility

- Results may vary by ±1–2 F1 points across runs due to LLM non-determinism at temperature > 0.
- The LoCoMo benchmark is known to contain approximately 6.4% erroneous ground-truth answers (see paper Section 4.2). Reported scores are therefore conservative lower bounds.
- The eval script uses `openai/gpt-4o-mini` by default (via OpenRouter). Change `MODEL` at the top of `eval_locomo.py` to use a different model.
- A-MEM comparison numbers in the paper (Table 1) are A-MEM's self-reported GPT-4o-mini results from their original paper. We did not run A-MEM ourselves. See the reproducibility caveat in Section 5 of the paper.

---

## Files

| File | Description |
|------|-------------|
| `eval_locomo.py` | Main evaluation script — runs full 10-conversation LoCoMo eval |
| `eval_longmemeval.py` | LongMemEval evaluation (in progress, not yet reported in paper) |
| `results_latest.jsonl` | Raw per-question results from the most recent run |
