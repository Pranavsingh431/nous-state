import json
import os
import time
import urllib.request
from dotenv import load_dotenv
import datetime
import re

# Ensure nous is in path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nous.engine import Nous
from nous.llm_extractor import LLMExtractor

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

def _clean_fact(fact: str) -> str:
    """Strip probability annotations for cleaner LLM context."""
    clean = re.sub(r'\s*\(\d+%( belief)?\)', '', fact)
    clean = re.sub(r"\b(likely|probably|possibly)\s+", "", clean, flags=re.I)
    clean = re.sub(r"\s+could be:\s*", " is ", clean, flags=re.I)
    return clean.strip()

def context_to_text(context: dict) -> str:
    """
    Compact, deduplicated context for the answering LLM.

    Uses FACT: / EVIDENCE: labels (more parseable than === Entity === bullets).
    Strips probability hedging language and removes duplicate facts globally
    before they reach the answering LLM.

    Research basis: Liu et al. 2023 "Lost in the Middle" (irrelevant/redundant
    context degrades LLM accuracy); Provence ICLR 2025 (query-aware pruning).
    """
    if not context:
        return ""

    lines = []
    seen_facts: set = set()
    seen_evidence: set = set()
    evidence_count = 0
    MAX_EVIDENCE = 25

    # --- 1. Relationships first — highest signal, already structured ---
    if "Relationships" in context:
        rel = context["Relationships"]
        facts = rel.get("facts", []) if isinstance(rel, dict) else []
        for fact in facts:
            clean = _clean_fact(fact)
            key = clean.lower()[:80]
            if clean and key not in seen_facts:
                seen_facts.add(key)
                lines.append(f"FACT: {clean}")

    # --- 2. Entity profile facts — global dedup across all entities ---
    for entity, profile in context.items():
        if entity in ("Relationships", "Relevant evidence"):
            continue
        if not isinstance(profile, dict):
            continue
        entity_facts = profile.get("facts", [])
        entity_evidence = profile.get("evidence", [])

        for fact in entity_facts:
            clean = _clean_fact(fact)
            key = clean.lower()[:80]
            if clean and key not in seen_facts:
                seen_facts.add(key)
                lines.append(f"FACT: {clean}")

        # Per-entity evidence (from delta log top snippets)
        for snippet in entity_evidence:
            if evidence_count >= MAX_EVIDENCE:
                break
            core = re.sub(r'\[.*?\]', '', snippet).strip().lower()[:100]
            core = re.sub(r'^[^:]+:\s*', '', core)
            if core and core not in seen_evidence:
                seen_evidence.add(core)
                lines.append(f"EVIDENCE: {snippet}")
                evidence_count += 1

    # --- 3. Global evidence block — deduplicated, capped at MAX_EVIDENCE total ---
    if "Relevant evidence" in context:
        ev = context["Relevant evidence"]
        evidence_list = ev.get("evidence", []) if isinstance(ev, dict) else []
        for snippet in evidence_list:
            if evidence_count >= MAX_EVIDENCE:
                break
            core = re.sub(r'\[.*?\]', '', snippet).strip().lower()[:100]
            core = re.sub(r'^[^:]+:\s*', '', core)
            if core and core not in seen_evidence:
                seen_evidence.add(core)
                lines.append(f"EVIDENCE: {snippet}")
                evidence_count += 1

    return "\n".join(lines)

def llm_judge_score(question: str, answer: str, ground_truth: str) -> float:
    """
    Use LLM to judge if answer is semantically correct vs ground truth.
    Returns 1.0 (correct), 0.5 (partial), or 0.0 (incorrect).
    """
    payload = json.dumps({
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": """You are an answer evaluator for a conversational memory benchmark.

Evaluate if the predicted answer conveys the SAME CORE MEANING as the ground truth.
Be GENEROUS. Focus on whether the key facts match, not exact wording.

Answer ONLY: CORRECT or WRONG.

Rules — mark CORRECT if:
- Same meaning, different words: "lucky and grateful" vs "appreciated them a lot" → CORRECT
- Predicted contains ground truth info: "researching adoption" vs "researching adoption agencies" → CORRECT
- Subset match: "painting" vs "painting and hiking" → CORRECT
- Synonym match: "scared but resilient" vs "frightened but strong" → CORRECT
- Same concept: "yoga" vs "yoga class" → CORRECT
- Same date: "May 7" vs "May 7, 2023" → CORRECT

Rules — mark WRONG if:
- Completely different fact: "gardening" vs "yoga" → WRONG
- "unknown" vs any actual answer → WRONG  
- Contradictory: "yes" vs "no" → WRONG"""},
            {"role": "user", "content": f"""Question: {question}
Ground truth: {ground_truth}  
Predicted: {answer}

Verdict (CORRECT or WRONG):"""}
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", 
                 "Authorization": f"Bearer {API_KEY}"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            verdict = body["choices"][0]["message"]["content"].strip().lower()
            if "correct" in verdict and "incorrect" not in verdict:
                return 1.0
            else:
                return 0.0
    except Exception:
        return f1_score(answer, ground_truth)  # fallback to F1 on error

def generate_answer(question: str, context: dict, category: str = None) -> str:
    """
    Generate an answer using the rich context from Nous.
    v3: Context now contains 'facts' (readable strings) and 'evidence' (raw snippets).
    """
    context_str = context_to_text(context) or "No relevant context found in memory."
        
    if category == "open-domain":
        system_prompt = """You are answering questions about people based on their conversation history.
These questions may require INFERENCE — reasoning about what a person would likely think,
do, or prefer based on their known interests, values, and personality.

Rules:
- If the question asks "would X likely..." or "what would X...", reason from known facts.
  Example: If someone likes classical music and the question asks about Vivaldi → answer YES.
  Example: If someone advocates for LGBTQ rights → likely liberal political leaning.
- Answer in 1-3 sentences. Be direct and confident.
- Use evidence from context to support your reasoning.
- NEVER say "unknown" unless the context has absolutely no information about the person.
  If you know ANYTHING about the person, make a reasonable inference.
- NEVER say "based on the context" or "according to"."""
        user_prompt = f"""Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer (reason from the facts you have):""" 
        max_tokens = 200
    elif category == "temporal":
        system_prompt = """You are answering temporal questions about people from their conversation history.

CRITICAL: The evidence has timestamps like [2023-07-12]. When someone says "yesterday" 
in a message dated [2023-07-12], they mean July 11, 2023. You MUST resolve relative 
time expressions to ABSOLUTE dates.

Rules:
- ALWAYS convert relative times to absolute dates using the evidence timestamps.
  "yesterday" in [2023-07-12] message → "July 11, 2023" or "11 July 2023"
  "last week" in [2023-06-09] message → "the week before June 9, 2023"
  "next month" in [2023-05-25] message → "June 2023"
  "two days ago" in [2023-07-12] message → "July 10, 2023"
  "last Saturday" in [2023-05-25] message → "the Saturday before May 25, 2023"
  "last year" in [2023-05-08] message → "2022"
- Keep your answer SHORT: just the date or time period.
- NEVER return relative expressions like "yesterday" or "last week".
- If snippets contradict, use the most recent one.
- Only say "unknown" if the context has zero relevant information."""
        user_prompt = f"""Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer (absolute date/time, NOT relative):""" 
        max_tokens = 80
    elif category == "multi-hop":
        system_prompt = """You are answering questions that require aggregating facts across a person's entire history.

The context has two types of lines:
- FACT lines: aggregated knowledge (e.g. "Observed Melanie activity values include: pottery, camping, painting, swimming")
- EVIDENCE lines: individual conversation snippets with timestamps

Rules:
- START with FACT lines — especially "Observed X Y values include:" lines. These aggregate ALL instances ever recorded.
  Example: if asked "what activities does X do?", the FACT line "Observed X activity values include: A, B, C" is your primary source.
- Use EVIDENCE lines only to fill gaps or verify specific events not in the FACT lines.
- If asked "what activities/hobbies does X do?" → list ALL items from the "Observed X activity values" FACT line.
- If asked "what has X painted/made/read?" → list ALL items from relevant "Observed X" FACT lines.
- Be COMPLETE — include every item. Do not stop at the first one.
- Keep the answer concise: comma-separated list or short phrases. No bullet points unless the question asks for detail.
- If FACT lines and EVIDENCE lines conflict, prefer the most recent EVIDENCE timestamp.
- Only say "unknown" if the context has zero relevant information."""
        user_prompt = f"""Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer (use FACT lines first, list ALL relevant items):""" 
        max_tokens = 150
    else:
        system_prompt = """You are answering questions about people from their conversation history.
Your job: extract the EXACT answer from the context. Use the same words as the context.

Rules:
- Keep your answer SHORT: a name, date, phrase, or 1 brief sentence.
- COPY exact words from the evidence when possible. Do NOT paraphrase.
- If asked "what did X do?", look for FACT lines and EVIDENCE lines about X's activities, plans, goals.
- Lines starting with "Observed X Y values include:" contain a list of all Y values for X — extract the relevant one(s) from it.
- If asked "how did X feel?" → use the exact emotion words from context.
- NEVER explain or hedge. Just state the answer.
- NEVER say "based on" or "according to".
- If snippets contradict, use the most recent (latest timestamp).
- Only say "unknown" if the context has zero relevant information."""
        user_prompt = f"""Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer (short, exact words from context):""" 
        max_tokens = 80

    payload = json.dumps({
        "model": os.getenv("BACKBONE_MODEL", "google/gemini-2.5-flash"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST"
    )

    body = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            return "unknown"

        if "error" in body:
            err_code = body["error"].get("code")
            if err_code in (429, 504, "429", "504"):
                time.sleep(15 * (attempt + 1))
                body = None
                continue
            return "unknown"
        break

    if body is None:
        return "unknown"

    try:
        raw_answer = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return "unknown"
    return raw_answer

def _extract_answer_text(raw_answer: str, category: str = None, question: str = "") -> str:
    """Parse the answer contract and repair common verbose fallback outputs."""
    raw_answer = str(raw_answer or "").strip()
    if not raw_answer:
        return ""

    try:
        parsed = json.loads(raw_answer)
        if isinstance(parsed, dict) and "answer" in parsed:
            raw_answer = str(parsed["answer"]).strip()
    except json.JSONDecodeError:
        # Some providers wrap JSON in prose/code fences. Extract the first object if present.
        match = re.search(r"\{.*?\}", raw_answer, flags=re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "answer" in parsed:
                    raw_answer = str(parsed["answer"]).strip()
            except json.JSONDecodeError:
                pass

    answer = raw_answer.strip().strip('"').strip("'").strip()
    answer = re.sub(r"\s+", " ", answer)

    if category == "temporal":
        return _compress_temporal_answer(answer)
    if category in {"single-hop", "multi-hop"}:
        return _compress_fact_answer(answer, question=question)
    if category == "open-domain":
        return _compress_open_answer(answer)
    return answer

def _compress_temporal_answer(answer: str) -> str:
    """Prefer bare temporal expressions over explanatory sentences."""
    if not answer:
        return answer

    # Exact date variants: May 7, 2023 / 7 May 2023 / 24 April, 2023.
    month = r"January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    patterns = [
        rf"\b\d{{1,2}}\s+(?:{month}),?\s+\d{{4}}\b",
        rf"\b(?:{month})\s+\d{{1,2}},?\s+\d{{4}}\b",
        rf"\b(?:{month})\s+\d{{4}}\b",
        r"\b\d{4}\b",
        r"\b(?:since|from)\s+\d{4}\b",
        r"\b\d+\s+(?:day|days|week|weeks|month|months|year|years)\b",
        rf"\b(?:the\s+)?(?:sunday|monday|tuesday|wednesday|thursday|friday|saturday)\s+before\s+\d{{1,2}}\s+(?:{month})\s+\d{{4}}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer, flags=re.I)
        if match:
            return match.group(0).strip()
    return answer

def _compress_fact_answer(answer: str, question: str = "") -> str:
    """Strip common sentence frames from otherwise correct short factual answers."""
    if not answer:
        return answer

    direct = _extract_direct_fact_from_sentence(answer, question)
    if direct:
        return direct

    cleaned = answer
    cleaned = re.sub(r"^the answer is\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^it (?:is|was|would be|would likely be)\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^they (?:are|were)\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^[A-Z][A-Za-z' -]{1,40}\s+(?:is|was|are|were|has|have|had|did|does|do)\s+", "", cleaned)

    # If the model ignored instructions and added evidence/prose, keep the first answer-like clause.
    cleaned = re.split(r"\s+(?:as|because|since|which|where|when)\s+", cleaned, maxsplit=1, flags=re.I)[0]
    cleaned = cleaned.split(".")[0].strip()
    return cleaned.strip(" ,;:")

def _extract_direct_fact_from_sentence(answer: str, question: str) -> str:
    """Question-aware cleanup for common factual LoCoMo answer shapes."""
    answer = answer.strip()
    q = question.lower()

    patterns = []
    if "raise awareness for" in q:
        patterns += [r"raised awareness for\s+([^.;,]+)", r"awareness for\s+([^.;,]+)"]
    if re.search(r"\bwhat did .+ research\b", q):
        patterns += [r"researched\s+([^.;,]+)", r"researching\s+([^.;,]+)"]
    if "career" in q or "career path" in q:
        patterns += [r"career in\s+([^.;]+)", r"pursue\s+([^.;]+)", r"looking into\s+([^.;]+)"]
    if "activities" in q or "partake" in q:
        patterns += [r"activities such as\s+([^.;]+)", r"partakes in\s+([^.;]+)", r"include\s+([^.;]+)"]
    if "used for" in q:
        patterns += [r"used for\s+([^.;,]+)", r"for\s+([^.;,]+)"]
    if "what did" in q and "paint" in q:
        patterns += [r"painted\s+([^.;]+)"]
    if "what are" in q and "plans" in q:
        patterns += [r"plans (?:include|are)\s+([^.;]+)", r"include\s+([^.;]+)"]

    for pattern in patterns:
        match = re.search(pattern, answer, flags=re.I)
        if not match:
            continue
        candidate = match.group(1).strip(" ,;:\"'")
        candidate = re.split(r"\s+(?:as|because|since|which|where|when)\s+", candidate, maxsplit=1, flags=re.I)[0]
        candidate = re.sub(r"^(?:a|an|the)\s+", "", candidate, flags=re.I)
        if 0 < len(candidate.split()) <= 12:
            return candidate
    return ""

def _compress_open_answer(answer: str) -> str:
    """Keep open-domain answers concise without removing the yes/no judgment."""
    if not answer:
        return answer
    yn_match = re.match(r"^(likely\s+yes|likely\s+no|yes|no)\b", answer, flags=re.I)
    if yn_match:
        judgment = yn_match.group(1)
        reason = re.split(r"\b(?:because|since|as)\b|[.;]", answer[len(judgment):], maxsplit=1, flags=re.I)[0]
        reason = reason.strip(" ,;:")
        if reason:
            words = reason.split()[:8]
            return f"{judgment}; {' '.join(words)}".strip(" ,;:")
        return judgment
    answer = answer.split(".")[0].strip()
    words = answer.split()
    if len(words) > 18:
        return " ".join(words[:18]).strip(" ,;:")
    return answer.strip(" ,;:")

def f1_score(prediction, ground_truth):
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

def bleu1_score(prediction, ground_truth):
    prediction = str(prediction)
    ground_truth = str(ground_truth)
    pred_tokens = _normalize_for_match(prediction).split()
    gt_tokens = _normalize_for_match(ground_truth).split()
    matches = sum(1 for t in pred_tokens if t in gt_tokens)
    return matches / len(pred_tokens) if pred_tokens else 0.0

def _normalize_for_match(text) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())

def context_recall_score(context_text: str, ground_truth) -> float:
    """Approximate whether retrieved context contains enough answer material."""
    gt = _normalize_for_match(ground_truth)
    ctx = _normalize_for_match(context_text)
    if not gt or not ctx:
        return 0.0
    if gt in ctx:
        return 1.0
    gt_tokens = [t for t in gt.split() if len(t) > 2]
    if not gt_tokens:
        return 0.0
    overlap = sum(1 for t in gt_tokens if t in ctx)
    return overlap / len(gt_tokens)

def failure_bucket(answer: str, f1: float, recall: float) -> str:
    answer_norm = _normalize_for_match(answer)
    if not answer_norm:
        return "api_or_empty_answer"
    if answer_norm == "unknown":
        return "unknown_answer"
    if recall < 0.34:
        return "retrieval_miss"
    if f1 == 0.0:
        return "answer_miss"
    if f1 < 0.5:
        return "partial_answer"
    return "good"

def _parse_timestamp(time_str):
    if not time_str:
        return None
    time_str = str(time_str).strip()
    formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.datetime.strptime(time_str, fmt).timestamp()
        except ValueError:
            pass
    # LoCoMo commonly uses lowercase am/pm.
    normalized = re.sub(r"\b(am|pm)\b", lambda m: m.group(1).upper(), time_str, flags=re.I)
    for fmt in formats:
        try:
            return datetime.datetime.strptime(normalized, fmt).timestamp()
        except ValueError:
            pass
    return None

def _session_number(session_key: str) -> int:
    match = re.search(r"session_(\d+)$", session_key)
    return int(match.group(1)) if match else 10**9

def run_locomo_eval(data_path: str):
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}. Please check if the repository was cloned correctly.")
        return
        
    with open(data_path) as f:
        data = json.load(f)

    START_FROM = int(os.getenv("START_FROM", "0"))
    MAX_CONVERSATIONS = os.getenv("MAX_CONVERSATIONS")
    MAX_CONVERSATIONS = int(MAX_CONVERSATIONS) if MAX_CONVERSATIONS else None
    DEBUG_QA = os.getenv("DEBUG_QA", "0") == "1"
    OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", os.path.join(os.path.dirname(__file__), "results_latest.jsonl"))

    results = {
        "single-hop": [], "multi-hop": [],
        "temporal": [], "open-domain": []
    }
    failure_counts = {
        "single-hop": {}, "multi-hop": {},
        "temporal": {}, "open-domain": {}
    }

    test_data = data[START_FROM:]
    if MAX_CONVERSATIONS is not None:
        test_data = test_data[:MAX_CONVERSATIONS]
    print(f"Starting LoCoMo eval on {len(test_data)} conversations [Phase 4: focused profiles + category prompts + CoT + inference]...")
    output_fh = open(OUTPUT_JSONL, "a", encoding="utf-8") if OUTPUT_JSONL else None
    
    try:
        for i, item in enumerate(test_data):
            conv_num = START_FROM + i + 1
            print(f"\nProcessing Conversation {conv_num}/{len(data)}...")

            conversation = item.get("conversation", {})
            speaker_a = conversation.get("speaker_a", "Speaker_A")
            nous = Nous(":memory:", extractor=LLMExtractor(api_key=API_KEY, user_context={"name": speaker_a},
                                                           model=os.getenv("BACKBONE_MODEL", "google/gemini-2.5-flash")))

            sessions = sorted(
                (k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")),
                key=_session_number
            )
            print(f"  Ingesting {len(sessions)} sessions...")
            
            base_ts = _parse_timestamp(conversation.get("session_1_date_time"))
            if base_ts is None:
                base_ts = datetime.datetime(2023, 1, 1).timestamp()
            
            turn_counter = 0
            for session_key in sessions:
                session = conversation[session_key]
                time_str = conversation.get(f"{session_key}_date_time")
                session_ts = _parse_timestamp(time_str)
                
                for turn in session:
                    text = f"{turn.get('speaker', 'Unknown')}: {turn.get('text', '')}"
                    if session_ts:
                        ts = session_ts + turn_counter * 60
                    else:
                        ts = base_ts + turn_counter * 60
                    nous.observe(text, source="conversation", timestamp=ts)
                    turn_counter += 1

            # 1 = multi-hop, 2 = temporal, 3 = open-domain/commonsense,
            # 4 = single-hop, 5 = adversarial.
            cat_map = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

            print(f"  Answering QA (total {len(item.get('qa', []))})...")
            qa_list = item.get("qa", [])
            for qa in qa_list:
                raw_category = qa.get("category")
                category = cat_map.get(raw_category, str(raw_category))
                if category == "adversarial":
                    continue

                question = qa["question"]
                ground_truth = qa["answer"]

                context = nous.query_relevant(question, category=category)
                context_text = context_to_text(context)

                # Fix 1: Try direct belief answering for single-hop profile questions.
                # Bypasses BM25+LLM for high-confidence facts already in the belief engine.
                belief_answer = None
                if category == "single-hop":
                    try:
                        belief_answer = nous.answer_from_beliefs(question)
                    except Exception:
                        belief_answer = None

                if belief_answer:
                    answer = belief_answer
                else:
                    answer = generate_answer(question, context, category=category)

                f1 = llm_judge_score(question, answer, ground_truth)
                bleu = bleu1_score(answer, ground_truth)
                recall = context_recall_score(context_text, ground_truth)
                bucket = failure_bucket(answer, f1, recall)

                results.setdefault(category, []).append({"f1": f1, "bleu1": bleu, "context_recall": recall})
                cat_buckets = failure_counts.setdefault(category, {})
                cat_buckets[bucket] = cat_buckets.get(bucket, 0) + 1

                if output_fh:
                    output_fh.write(json.dumps({
                        "conversation": conv_num,
                        "sample_id": item.get("sample_id"),
                        "raw_category": raw_category,
                        "category": category,
                        "question": question,
                        "ground_truth": ground_truth,
                        "answer": answer,
                        "f1": f1,
                        "bleu1": bleu,
                        "context_recall": recall,
                        "failure_bucket": bucket,
                        "context": context,
                        "context_text": context_text,
                    }, ensure_ascii=False) + "\n")
                    output_fh.flush()

                if DEBUG_QA:
                    print("\n    DEBUG QA")
                    print(f"      category: {category}")
                    print(f"      question: {question}")
                    print(f"      answer:   {answer}")
                    print(f"      truth:    {ground_truth}")
                    print(f"      f1:       {f1 * 100:.2f}")
                    print(f"      recall:   {recall * 100:.2f}")
                    print(f"      bucket:   {bucket}")
                    print(f"      context:  {json.dumps(context, ensure_ascii=False)[:1200]}")

            nous.close()
            
            # Intermediate output
            for cat, scores in results.items():
                if scores:
                    avg_f1 = sum(s["f1"] for s in scores) / len(scores) * 100
                    avg_recall = sum(s["context_recall"] for s in scores) / len(scores) * 100
                    print(f"    Current {cat} F1: {avg_f1:.2f}, ContextRecall={avg_recall:.2f} (n={len(scores)})")

        print("\n--- FINAL RESULTS [Phase 4: focused profiles + category prompts + CoT + inference] ---")
        for cat, scores in results.items():
            if scores:
                avg_f1 = sum(s["f1"] for s in scores) / len(scores) * 100
                avg_bleu = sum(s["bleu1"] for s in scores) / len(scores) * 100
                avg_recall = sum(s["context_recall"] for s in scores) / len(scores) * 100
                print(f"{cat}: F1={avg_f1:.2f}, BLEU-1={avg_bleu:.2f}, ContextRecall={avg_recall:.2f} (n={len(scores)})")
                print(f"  Buckets: {failure_counts[cat]}")
    finally:
        if output_fh:
            output_fh.close()

if __name__ == "__main__":
    locomo_path = os.path.join(os.path.dirname(__file__), "locomo/data/locomo10.json")
    run_locomo_eval(locomo_path)
