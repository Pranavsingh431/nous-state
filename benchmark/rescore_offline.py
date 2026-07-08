"""
Offline re-scorer for Nous LoCoMo results.

Recomputes TRUE strict token-level F1 and BLEU-1 from a saved results JSONL
(the per-question dump written by eval_locomo.py), with ZERO API calls.

Why this exists:
    eval_locomo.py reports `llm_judge_score` (a generous GPT-4o-mini judge)
    under the variable name `f1`. The paper claims strict token-level F1.
    This script computes the *actual* strict token-F1 offline so we can see
    what the honest numbers are without re-running the 7-8h eval.

The scoring functions below are copied verbatim from eval_locomo.py so the
recomputed BLEU-1 exactly reproduces the saved `bleu1` field (a self-check
that we are reading the right answer/ground_truth fields).

Usage:
    python benchmark/rescore_offline.py [path/to/results_latest.jsonl]
"""

import json
import os
import re
import sys


# ---- Scoring, copied verbatim from eval_locomo.py -------------------------

def _normalize_for_match(text) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def f1_score(prediction, ground_truth) -> float:
    prediction = str(prediction)
    ground_truth = str(ground_truth)
    pred_tokens = _normalize_for_match(prediction).split()
    gt_tokens = _normalize_for_match(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(prediction, ground_truth) -> float:
    prediction = str(prediction)
    ground_truth = str(ground_truth)
    pred_tokens = _normalize_for_match(prediction).split()
    gt_tokens = _normalize_for_match(ground_truth).split()
    matches = sum(1 for t in pred_tokens if t in gt_tokens)
    return matches / len(pred_tokens) if pred_tokens else 0.0


# ---- Aggregation ----------------------------------------------------------

CATEGORY_ORDER = ["single-hop", "multi-hop", "temporal", "open-domain"]


def rescore(path: str) -> None:
    if not os.path.exists(path):
        print(f"[!] Results file not found: {path}")
        print("    Point this script at the results_latest.jsonl from the run")
        print("    that produced the paper numbers.")
        sys.exit(1)

    per_cat = {}   # category -> list of dicts
    n_rows = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cat = row.get("category", "?")
            answer = row.get("answer", "")
            gt = row.get("ground_truth", "")

            true_f1 = f1_score(answer, gt)
            true_bleu = bleu1_score(answer, gt)
            saved_judge = row.get("f1")        # this is the LLM-judge score (0/0.5/1)
            saved_bleu = row.get("bleu1")

            per_cat.setdefault(cat, []).append({
                "true_f1": true_f1,
                "true_bleu": true_bleu,
                "saved_judge": saved_judge,
                "saved_bleu": saved_bleu,
            })
            n_rows += 1

    print(f"\nRe-scored {n_rows} questions from {path}\n")

    header = f"{'Category':<14}{'n':>6}{'JUDGE (paper F1)':>18}{'TRUE token-F1':>16}{'TRUE BLEU-1':>14}{'saved BLEU-1':>14}"
    print(header)
    print("-" * len(header))

    cats = [c for c in CATEGORY_ORDER if c in per_cat] + \
           [c for c in per_cat if c not in CATEGORY_ORDER]

    macro_judge, macro_truef1, n_cats = 0.0, 0.0, 0
    for cat in cats:
        rows = per_cat[cat]
        n = len(rows)
        judge = _avg(r["saved_judge"] for r in rows) * 100
        truef1 = _avg(r["true_f1"] for r in rows) * 100
        truebleu = _avg(r["true_bleu"] for r in rows) * 100
        savedbleu = _avg(r["saved_bleu"] for r in rows) * 100
        print(f"{cat:<14}{n:>6}{judge:>17.2f}{truef1:>16.2f}{truebleu:>14.2f}{savedbleu:>14.2f}")
        if cat in CATEGORY_ORDER:
            macro_judge += judge
            macro_truef1 += truef1
            n_cats += 1

    if n_cats:
        print("-" * len(header))
        print(f"{'Macro avg':<14}{'':>6}{macro_judge / n_cats:>17.2f}{macro_truef1 / n_cats:>16.2f}")

    print("\nNotes:")
    print("  * 'JUDGE (paper F1)' = the number your paper currently reports as F1.")
    print("  * 'TRUE token-F1'    = the strict metric the paper *claims* to use.")
    print("  * If 'saved BLEU-1' != 'TRUE BLEU-1', the answer/GT fields were read wrong.")


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(__file__), "results_latest.jsonl")
    rescore(sys.argv[1] if len(sys.argv) > 1 else default)
