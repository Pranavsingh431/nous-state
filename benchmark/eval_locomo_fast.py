"""
Fast LoCoMo eval — parallelized, behaviour-identical to eval_locomo.py.

Speedups (no change to scoring, prompts, or belief semantics):
  1. Ingestion extraction is parallelized across turns (network-bound, order-independent),
     then Bayesian updates are applied strictly in original order — so belief evolution is
     identical to serial ingestion.
  2. Retrieval (query_relevant / answer_from_beliefs) runs serially in the main thread
     (fast, pure-CPU when dense is off, and avoids any thread-safety questions).
  3. Only the stateless network calls (generate_answer, llm_judge_score) are parallelized.

All scoring functions, prompts, models, and the output JSONL schema are imported from
eval_locomo.py unchanged, so benchmark/rescore_offline.py works on the output as-is.

Env:
  MAX_CONVERSATIONS, START_FROM, OUTPUT_JSONL  (same as eval_locomo.py)
  INGEST_WORKERS (default 16), QA_WORKERS (default 8)
  NOUS_ABLATE_UPDATE / NOUS_ABLATE_AGG / NOUS_ABLATE_DENSE  (ablation toggles)
"""

import json
import os
import sys
import datetime
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nous.engine import Nous
from nous.llm_extractor import LLMExtractor

# Reuse everything from the canonical eval so behaviour is identical.
import eval_locomo as E

INGEST_WORKERS = int(os.getenv("INGEST_WORKERS", "16"))
QA_WORKERS = int(os.getenv("QA_WORKERS", "8"))

CAT_MAP = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def _safe_extract(extractor, text):
    try:
        return extractor.extract(text)
    except Exception:
        return []


def run(data_path):
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    with open(data_path) as f:
        data = json.load(f)

    START_FROM = int(os.getenv("START_FROM", "0"))
    MAX = os.getenv("MAX_CONVERSATIONS")
    MAX = int(MAX) if MAX else None
    OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", os.path.join(os.path.dirname(__file__), "results_fast.jsonl"))

    test_data = data[START_FROM:]
    if MAX is not None:
        test_data = test_data[:MAX]

    print(f"[fast] LoCoMo eval on {len(test_data)} conversation(s) | "
          f"ingest_workers={INGEST_WORKERS} qa_workers={QA_WORKERS} | "
          f"ablate update={os.getenv('NOUS_ABLATE_UPDATE','-') or '-'} "
          f"agg={os.getenv('NOUS_ABLATE_AGG','0')} dense={os.getenv('NOUS_ABLATE_DENSE','0')}", flush=True)

    results = {"single-hop": [], "multi-hop": [], "temporal": [], "open-domain": []}
    out_fh = open(OUTPUT_JSONL, "a", encoding="utf-8")

    try:
        for i, item in enumerate(test_data):
            conv_num = START_FROM + i + 1
            conversation = item.get("conversation", {})
            speaker_a = conversation.get("speaker_a", "Speaker_A")
            nous = Nous(":memory:", extractor=LLMExtractor(api_key=E.API_KEY, user_context={"name": speaker_a}))

            sessions = sorted(
                (k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")),
                key=E._session_number,
            )

            base_ts = E._parse_timestamp(conversation.get("session_1_date_time")) or \
                datetime.datetime(2023, 1, 1).timestamp()

            # --- Gather all turns in order, with timestamps ---
            turns = []  # (text, ts)
            counter = 0
            for session_key in sessions:
                session = conversation[session_key]
                session_ts = E._parse_timestamp(conversation.get(f"{session_key}_date_time"))
                for turn in session:
                    text = f"{turn.get('speaker', 'Unknown')}: {turn.get('text', '')}"
                    ts = (session_ts if session_ts else base_ts) + counter * 60
                    turns.append((text, ts))
                    counter += 1

            # --- 1. PARALLEL extraction (order preserved) ---
            extractor = nous.extractor
            print(f"[conv {conv_num}] extracting {len(turns)} turns "
                  f"across {INGEST_WORKERS} workers...", flush=True)
            with ThreadPoolExecutor(max_workers=INGEST_WORKERS) as ex:
                claims_list = list(ex.map(lambda tx: _safe_extract(extractor, tx),
                                          [t[0] for t in turns]))

            # --- 2. SEQUENTIAL ordered belief updates ---
            for (text, ts), claims in zip(turns, claims_list):
                nous.observe(text, source="conversation", timestamp=ts, claims=claims)

            # --- 3. Retrieval (serial, fast) then parallel answer+judge ---
            qa_list = [qa for qa in item.get("qa", []) if CAT_MAP.get(qa.get("category")) != "adversarial"]
            print(f"[conv {conv_num}] answering {len(qa_list)} questions "
                  f"across {QA_WORKERS} workers...", flush=True)

            prepared = []
            for qa in qa_list:
                category = CAT_MAP.get(qa.get("category"), str(qa.get("category")))
                question = qa["question"]
                context = nous.query_relevant(question, category=category)
                context_text = E.context_to_text(context)
                belief_answer = None
                if category == "single-hop":
                    try:
                        belief_answer = nous.answer_from_beliefs(question)
                    except Exception:
                        belief_answer = None
                prepared.append((qa, category, context, context_text, belief_answer))

            def _finish(p):
                qa, category, context, context_text, belief_answer = p
                question = qa["question"]
                ground_truth = qa["answer"]
                answer = belief_answer or E.generate_answer(question, context, category=category)
                f1 = E.llm_judge_score(question, answer, ground_truth)   # judge (paper metric)
                bleu = E.bleu1_score(answer, ground_truth)
                recall = E.context_recall_score(context_text, ground_truth)
                bucket = E.failure_bucket(answer, f1, recall)
                return {
                    "conversation": conv_num, "sample_id": item.get("sample_id"),
                    "raw_category": qa.get("category"), "category": category,
                    "question": question, "ground_truth": ground_truth, "answer": answer,
                    "f1": f1, "bleu1": bleu, "context_recall": recall,
                    "failure_bucket": bucket, "context": context, "context_text": context_text,
                }

            with ThreadPoolExecutor(max_workers=QA_WORKERS) as ex:
                rows = list(ex.map(_finish, prepared))

            for row in rows:
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                results.setdefault(row["category"], []).append(
                    {"f1": row["f1"], "bleu1": row["bleu1"], "context_recall": row["context_recall"]})
            out_fh.flush()
            nous.close()

            for cat, scores in results.items():
                if scores:
                    avg = sum(s["f1"] for s in scores) / len(scores) * 100
                    print(f"    {cat}: judge={avg:.2f} (n={len(scores)})")

        print("\n--- FINAL RESULTS [fast] ---")
        for cat, scores in results.items():
            if scores:
                avg = sum(s["f1"] for s in scores) / len(scores) * 100
                print(f"{cat}: judge-F1={avg:.2f} (n={len(scores)})")
    finally:
        out_fh.close()


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "locomo/data/locomo10.json")
    run(path)
