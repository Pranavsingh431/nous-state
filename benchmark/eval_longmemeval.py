import json
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Ensure nous is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nous.engine import Nous

# Import helper functions directly from eval_locomo
from benchmark.eval_locomo import (
    generate_answer,
    f1_score,
    bleu1_score
)

load_dotenv()

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/longmemeval_s_cleaned.json"
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run download_longmemeval.sh first.")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Category mapping
    TYPE_MAP = {
        "single-session-user":        "single-session",
        "single-session-assistant":   "single-session", 
        "single-session-preference":  "single-session",
        "temporal-reasoning":         "temporal",
        "knowledge-update":           "knowledge-update",
        "multi-session":              "multi-session",
    }

    results = {
        "single-session": [],
        "temporal": [],
        "knowledge-update": [],
        "multi-session": [],
        "abstention": [],
    }

    total_items = len(data)
    for i, item in enumerate(data):
        q_id = item.get("question_id", "")
        raw_type = item.get("question_type", "")
        question = item.get("question", "")
        ground_truth = item.get("answer", "")
        haystack_sessions = item.get("haystack_sessions", [])
        haystack_dates = item.get("haystack_dates", [])

        is_abstention = str(q_id).endswith("_abs")
        
        category = "abstention" if is_abstention else TYPE_MAP.get(raw_type, raw_type)
        
        if category not in results:
            results[category] = []

        print(f"Processing item {i+1}/{total_items} [{category}]...")

        # Create a fresh Nous instance in memory for each item
        nous = Nous(":memory:")

        # Ingestion loop
        for s_idx, session in enumerate(haystack_sessions):
            date_str = haystack_dates[s_idx] if s_idx < len(haystack_dates) else ""
            
            # Parse timestamp
            try:
                if date_str:
                    dt = datetime.fromisoformat(date_str)
                    ts = dt.timestamp()
                else:
                    ts = time.time()
            except ValueError:
                ts = time.time()

            # Process turns
            for turn in session:
                if turn.get("role") == "user":
                    content = turn.get("content", "")
                    try:
                        nous.observe(content, source="user", timestamp=ts)
                    except Exception as e:
                        print(f"  Warning: failed to observe turn: {e}")
            
        # Pre-warm embedding cache
        print("  Pre-warming embedding cache...")
        try:
            nous.embed_all_evidence()
        except Exception as e:
            print(f"  Warning: failed to pre-warm embeddings: {e}")

        # QA Answering
        try:
            # Reusing exact logic from eval_locomo.py - query_relevant
            context = nous.query_relevant(question)
            
            answer = generate_answer(question, context, category=category)
        except Exception as e:
            print(f"  Warning: failed to generate answer: {e}")
            answer = ""

        # Scoring
        if is_abstention:
            ans_lower = answer.lower()
            if any(phrase in ans_lower for phrase in ["unknown", "don't know", "do not know", "no information"]):
                correct = True
            else:
                correct = False
            results["abstention"].append({"correct": correct, "answer": answer, "ground_truth": ground_truth})
        else:
            try:
                f1 = f1_score(answer, ground_truth)
                bleu = bleu1_score(answer, ground_truth)
                results[category].append({"f1": f1, "bleu1": bleu, "answer": answer, "ground_truth": ground_truth})
            except Exception as e:
                print(f"  Warning: failed to score: {e}")

        # Print running averages every 50 items
        if (i + 1) % 50 == 0:
            print(f"\n--- Running Averages at {i+1} ---")
            for cat, scores in results.items():
                n = len(scores)
                if n == 0:
                    continue
                if cat == "abstention":
                    acc = sum(1 for s in scores if s["correct"]) / n * 100
                    print(f"  {cat}: Accuracy={acc:.2f}% (n={n})")
                else:
                    avg_f1 = sum(s["f1"] for s in scores) / n
                    avg_bleu = sum(s["bleu1"] for s in scores) / n
                    print(f"  {cat}: F1={avg_f1:.2f}, BLEU-1={avg_bleu:.2f} (n={n})")
            print("---------------------------------\n")

    # Final Output
    print("\n--- FINAL RESULTS (LongMemEval-S) ---")
    overall_f1_sum = 0.0
    overall_f1_count = 0

    for cat in ["single-session", "temporal", "knowledge-update", "multi-session"]:
        scores = results.get(cat, [])
        n = len(scores)
        if n > 0:
            avg_f1 = sum(s["f1"] for s in scores) / n
            avg_bleu = sum(s["bleu1"] for s in scores) / n
            overall_f1_sum += sum(s["f1"] for s in scores)
            overall_f1_count += n
            print(f"{cat:<18} F1={avg_f1:.2f}, BLEU-1={avg_bleu:.2f} (n={n})")
        else:
            print(f"{cat:<18} F1=0.00, BLEU-1=0.00 (n=0)")

    abs_scores = results.get("abstention", [])
    abs_n = len(abs_scores)
    if abs_n > 0:
        abs_acc = sum(1 for s in abs_scores if s["correct"]) / abs_n * 100
        print(f"\nabstention:       Accuracy={abs_acc:.2f}% (n={abs_n})")
    else:
        print("\nabstention:       Accuracy=0.00% (n=0)")

    if overall_f1_count > 0:
        overall_f1 = overall_f1_sum / overall_f1_count
        print(f"\nOVERALL F1:       {overall_f1:.2f} (excluding abstention)")

    # Save to JSON
    out_path = "benchmark/results_longmemeval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
