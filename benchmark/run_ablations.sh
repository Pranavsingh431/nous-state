#!/usr/bin/env bash
#
# Controlled ablation suite for Nous on a LoCoMo subset — FAST (parallel) eval.
#
# Uses eval_locomo_fast.py (parallel extraction + parallel answer/judge; belief
# updates stay strictly ordered, so results are behaviour-identical to serial).
# Every variant is re-scored on BOTH metrics (LLM-judge AND true token-F1).
#
# Variants (all on the SAME fixed subset for a controlled comparison):
#   bm25_full    BM25 + BFS, full Bayesian engine        <- primary baseline
#                (BM25+BFS is the retrieval the paper actually describes)
#   bm25_flat    + Bayesian update -> naive overwrite     (tests belief mechanism)
#   bm25_noagg   + observed-values aggregation disabled   (tests multi-hop agg)
#   dense_full   dense retriever ON                       (measures dense's contribution
#                                                          vs bm25_full; slower/flaky, runs last)
#
# Usage:
#   MAX_CONVERSATIONS=3 bash benchmark/run_ablations.sh
#
# Requires OPENROUTER_API_KEY in .env (read by the eval).

set -uo pipefail
cd "$(dirname "$0")/.."

export MAX_CONVERSATIONS="${MAX_CONVERSATIONS:-3}"
export START_FROM="${START_FROM:-0}"
export INGEST_WORKERS="${INGEST_WORKERS:-16}"
export QA_WORKERS="${QA_WORKERS:-8}"

echo "Ablation suite (fast) | subset: first ${MAX_CONVERSATIONS} conv | workers ${INGEST_WORKERS}/${QA_WORKERS}"
echo "start: $(date)"
echo "========================================================================"

run_variant () {
  local name="$1"; shift
  local out="benchmark/ablation_${name}.jsonl"
  rm -f "$out"
  local t0=$(date +%s)
  echo ""
  echo ">>> VARIANT ${name}  (env: $*)  start $(date)"
  env "$@" OUTPUT_JSONL="$out" python -u benchmark/eval_locomo_fast.py
  local t1=$(date +%s)
  echo ">>> ${name} finished in $(( (t1 - t0) / 60 ))m$(( (t1 - t0) % 60 ))s"
  echo ">>> re-score ${name} (both metrics):"
  python benchmark/rescore_offline.py "$out"
}

run_variant bm25_full   NOUS_ABLATE_DENSE=1
run_variant bm25_flat   NOUS_ABLATE_DENSE=1 NOUS_ABLATE_UPDATE=flat
run_variant bm25_noagg  NOUS_ABLATE_DENSE=1 NOUS_ABLATE_AGG=1
run_variant dense_full  # dense ON (no ablation flags)

echo ""
echo "========================================================================"
echo "Ablation suite complete: $(date)"
echo "Compare bm25_flat vs bm25_full  -> does the Bayesian belief mechanism help?"
echo "Compare bm25_noagg vs bm25_full -> does observed-values aggregation help (multi-hop)?"
echo "Compare dense_full vs bm25_full -> does the (undocumented) dense retriever help?"
