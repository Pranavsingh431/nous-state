# Nous Ablations

The paper's own Limitations section names controlled ablations as *"the single
highest-priority extension to this work."* This is that harness. Every toggle defaults
OFF, so with no env vars set the system is byte-identical to the production/paper pipeline.

## What each ablation tests

| Variant | Env var | What it disables | Question it answers |
|---|---|---|---|
| `production` | (none) | nothing | full-system baseline on the subset |
| `flat_update` | `NOUS_ABLATE_UPDATE=flat` | Bayesian posterior → naive last-write-wins overwrite | **Does belief-tracking actually help, vs. just storing the latest value?** (the core claim) |
| `no_agg` | `NOUS_ABLATE_AGG=1` | observed-values delta aggregation (Eq. 7) → current best belief only | Does multi-value aggregation drive multi-hop performance? |
| `bm25_only` | `NOUS_ABLATE_DENSE=1` | dense embedding retriever → BM25 + 2-hop BFS only | How much does the (paper-undocumented) dense retrieval contribute vs. the "BM25 + BFS" the paper describes? |

**Note on entropy decay:** `apply_decay()` is *never called during the LoCoMo eval*
(`engine.py`), so decay contributes nothing to the benchmark numbers and is deliberately
**not** in this suite. Its value must be shown with a separate staleness test, not LoCoMo.

## Interpreting results — the honest read

For each variant the runner prints both **JUDGE** (the paper's reported metric) and
**TRUE token-F1**. The key comparisons:

- **`flat_update` vs `production`.** If the score drops sharply, the Bayesian belief
  mechanism is doing real work (compare BeliefMem, whose analogous ablation dropped F1
  42→20). If it barely moves, the "predictive world model" is *not* the source of value and
  the paper's central claim must be reframed. **We run this to learn, not to confirm.**
- **`bm25_only` vs `production`.** If this drops a lot, dense retrieval — undocumented in the
  paper — is carrying the result, which undercuts the "principled engine needs no embeddings"
  positioning and must be disclosed.
- **`no_agg` vs `production`, on multi-hop specifically.** Isolates the aggregation mechanism.

## Running

```bash
# from repo root; key is read from .env by the eval
MAX_CONVERSATIONS=3 bash benchmark/run_ablations.sh
```

Runs all four variants on the same fixed subset (first N conversations) and re-scores each.
Each variant ≈ 1.5–2.5h on 3 conversations. Outputs: `benchmark/ablation_<variant>.jsonl`.

Re-score any run offline at any time (free, no API):

```bash
python benchmark/rescore_offline.py benchmark/ablation_flat_update.jsonl
```
