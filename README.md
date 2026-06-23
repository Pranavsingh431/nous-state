# nous-state

**A probabilistic agent state layer for long-running personal AI agents.**

> "Knowledge is prediction, not storage."

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/nous-state)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2606.22030-b31b1b.svg)](https://arxiv.org/abs/2606.22030)

---

## Paper

This repository accompanies the preprint:

**Nous: A Predictive World Model for Long-Term Agent Memory**
Pranav Singh — Indian Institute of Technology Ropar
[arXiv:2606.22030](https://arxiv.org/abs/2606.22030) · cs.AI · June 2026

If you use this work, please cite:

```bibtex
@article{singh2026nous,
  title   = {Nous: A Predictive World Model for Long-Term Agent Memory},
  author  = {Singh, Pranav},
  journal = {arXiv preprint arXiv:2606.22030},
  year    = {2026}
}
```

To reproduce the LoCoMo benchmark results reported in the paper, see [`benchmark/README.md`](benchmark/README.md).

---

## The Problem

Every long-running AI agent eventually hits the same wall:

```
User (Month 1): "I work at Sarvam AI using Mistral for NyayaSahayak."
User (Month 4): "I switched to GPT-4 and joined Google DeepMind."
Agent (Month 5): *confidently tells someone Pranav uses Mistral at Sarvam AI*
```

Vector databases store both facts. Knowledge graphs require manual conflict resolution. Neither gives you a mathematically principled answer to *which fact is currently true*.

**nous-state** solves this with Bayesian probability distributions — the same math used in GPS navigation, spam filters, and medical diagnostics.

---

## How It Works

Instead of storing facts, Nous maintains belief distributions over entity attributes:

```python
Pranav.employer = { "Sarvam AI": 0.82, "unknown": 0.18 }
```

When new evidence arrives, it performs a Bayesian update:

```
"Pranav joined Google DeepMind" →
Pranav.employer = { "Google DeepMind": 0.86, "Sarvam AI": 0.12, "unknown": 0.02 }
```

Every update is recorded as an immutable **delta** — a change in understanding, not just a fact. This means:

- **Contradictions are resolved mathematically**, not heuristically
- **History is queryable** — "What did the agent believe about Pranav in March?"
- **Forgetting is principled** — unused beliefs decay toward uncertainty via entropy decay
- **Identity hints** — two entity names with high mutual information are flagged as likely the same person

---

## Install

```bash
pip install nous-state
```

Zero runtime dependencies. Pure Python stdlib only (`math`, `sqlite3`, `json`, `urllib`).

> ⚠️ **Honest caveat:** The engine's math is only as good as the extraction feeding it.
> The built-in rule-based extractor handles simple patterns. For messy real-world text,
> use the `LLMExtractor` — or bring your own parser. Garbage extraction → garbage beliefs,
> no matter how elegant the Bayesian update.

---

## Quickstart

### Rule-based extraction (no LLM needed)

```python
from nous import Nous

memory = Nous("agent_memory.db")

# Session 1
memory.observe("Pranav works at Sarvam AI as an ML engineer.")
memory.observe("He is building NyayaSahayak using Mistral.")

# Session 4 — things changed
memory.observe("Pranav left Sarvam AI and joined Google DeepMind.")
memory.observe("NyayaSahayak now uses GPT-4 for better legal reasoning.")

# Query current beliefs
print(memory.predict("Pranav", "employer"))
# → {"Google DeepMind": 0.86, "Sarvam AI": 0.12}

print(memory.predict("NyayaSahayak", "model"))
# → {"GPT-4": 0.86, "Mistral": 0.12}
```

### With LLM extraction (natural language → structured beliefs)

```python
from nous import Nous
from nous.llm_extractor import LLMExtractor

extractor = LLMExtractor(
    api_key="sk-or-...",            # Any OpenRouter key
    user_context={"name": "Pranav"} # Resolves "I/me/my" → "Pranav"
)

memory = Nous("agent_memory.db", extractor=extractor)

memory.observe("I switched from Mistral to GPT-4 because legal reasoning improved.")
memory.observe("Actually wait, someone said Pranav is still at Sarvam AI?")
memory.observe("No confirmed, he's definitely at Google DeepMind on Gemini.")

print(memory.predict("Pranav", "employer"))
# → {"Google DeepMind": 0.97, "Sarvam AI": 0.03}
```

### Explainability

```python
# Full auditable history
for delta in memory.history("Pranav", "employer"):
    print(f"Surprise: {delta.surprise:.1f} bits | {delta.evidence[:60]}")
# → Surprise: 4.3 bits | Actually, I left Sarvam AI and joined Google...
# → Surprise: 0.2 bits | Pranav is definitely at Google DeepMind, I saw...

# Time-travel: what did the agent believe 30 days ago?
past_belief = memory.query_at("Pranav", "employer", at_time=timestamp_30_days_ago)
```

### Surprise scoring

```python
bits = memory.surprise("Pranav is still at Sarvam AI.")
# → 5.1 bits  (high — contradicts current belief)

bits = memory.surprise("Pranav works at Google DeepMind.")
# → 0.1 bits  (low — we already know this)
```

---

## API Reference

### `Nous(db_path, extractor=None)`

| Method | Description |
|--------|-------------|
| `observe(text, source, reliability)` | Ingest text, update beliefs |
| `predict(entity, attribute)` | Get current probability distribution |
| `query_at(entity, attribute, timestamp)` | Time-travel query |
| `history(entity, attribute)` | Full delta log for an attribute |
| `explain(entity, attribute, value)` | Why does the agent believe X? |
| `surprise(text)` | Information content in bits before observing |
| `get_coupling(entity_a, entity_b)` | Identity similarity hint (0–1) |
| `get_entity_profile(entity)` | All attributes for an entity |
| `apply_decay(current_time)` | Apply forgetting to stale dimensions |

### `LLMExtractor(api_key, model, user_context)`

Works with any OpenAI-compatible API endpoint (OpenRouter, OpenAI, local via LM Studio).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | required | Your API key |
| `model` | `google/gemini-2.5-flash` | Any model on OpenRouter |
| `user_context` | `{}` | Dict with `name` key to resolve "I/me/my" |

---

## Architecture

```
Natural Language
      ↓
LLMExtractor (or rule-based Extractor)
      ↓
(entity, attribute, value) tuples
      ↓
BayesianUpdater → surprise score → posterior distribution
      ↓
Dimension (probability distribution)   +   Delta (immutable history)
      ↓                                          ↓
WorldModel (in-memory cache)          DeltaLog (SQLite)
      ↓
PersistenceLayer (SQLite — survives restarts)
```

**Key properties:**
- O(1) per-attribute reads — dictionary lookup, not vector search
- O(k) writes — multiply k floats, normalise (k = number of known values, typically < 10)
- Append-only delta log
- Zero external dependencies

---

## Where Nous Fits (and Where It Doesn't)

| Problem | Vector DB | Knowledge Graph | nous-state |
|---------|-----------|-----------------|------------|
| Contradictory facts | Stores both, LLM decides | Manual conflict rules | Bayesian update (automatic) |
| Stale high-confidence facts | Still retrieved | Still in graph | Probability mass shifts |
| "Why does agent believe X?" | Not possible | Requires audit log | Native (delta history) |
| Multi-hop relational queries | ❌ | ✅ Native | ❌ |
| Semantic document retrieval | ✅ Native | ❌ | ❌ |
| Per-attribute read cost | O(n) ANN search | O(edges) traversal | O(1) dict lookup |

---

## Research Status

This is research-stage code. The preprint ([arXiv:2606.22030](https://arxiv.org/abs/2606.22030)) describes this
as a first public report — not a final or fully audited result. Ablations, a second benchmark
(LongMemEval), and broader backbone evaluation are planned as immediate next steps and will be
reported in a future revision.

The `benchmark/` directory contains the evaluation code that produced the LoCoMo numbers in
the paper. See [`benchmark/README.md`](benchmark/README.md) for reproduction instructions.

---

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Issues and PRs welcome. If you hit a real-world edge case, opening an issue is genuinely valuable.
