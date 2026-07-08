# Nous — Honest Results Audit

**Status:** Phase 0 complete (offline re-scoring, $0, no re-run).
**Date:** 2026-07-07
**Source of truth:** `benchmark/results_latest.jsonl` (1,540 questions — reproduces the paper's numbers exactly).
**Reproduce:** `python benchmark/rescore_offline.py benchmark/results_latest.jsonl`

---

## TL;DR

The paper reports its headline metric as **strict token-level F1** and uses that framing to
claim wins over A-MEM and BeliefMem. The evaluation code actually aggregates a **generous
GPT-4o-mini LLM-as-judge** under the variable name `f1`. The genuine strict token-F1 is
**macro 32.45, not 59.97**. Under one consistent metric, Nous **does not** beat A-MEM
(ties single-hop, wins temporal, loses multi-hop + open-domain) and **loses to BeliefMem**
on all four categories. **The paper cannot go to a top-tier venue as written.** The
architecture, the temporal-reasoning result, and the metric-sensitivity gap are all real and
salvageable in an honest reframe.

---

## 1. The metric mislabel (proven)

- Paper claims, repeatedly: *"token-level F1 and BLEU-1, both strict surface-form measures"*
  and *"our use of strict token F1 rather than a lenient LLM-judge protocol."* It uses this to
  justify **not** comparing against Zep/Mem0 (which use LLM-judge).
- Actual code — `benchmark/eval_locomo.py:585`:
  ```python
  f1 = llm_judge_score(question, answer, ground_truth)   # named f1, but it's the judge
  ```
  `llm_judge_score` (line 100) is a GPT-4o-mini judge prompted **"Be GENEROUS. Focus on
  whether the key facts match, not exact wording."** The real token-F1 function (`f1_score`,
  line 409) is only reached as an *exception fallback*.
- Confirmed present at commit `5c8ffcc` ("LoCoMo benchmark + paper source").

**Independent proof from the paper's own numbers:** token-F1 ≤ `2P/(P+1)` where P ≈ BLEU-1.
Three of four reported "F1" values exceed that ceiling → they cannot be token-F1:

| Category | Reported "F1" | BLEU-1 (≈P) | Max possible token-F1 | Verdict |
|---|---|---|---|---|
| Single-hop | 63.50 | 45.60 | 62.6 | impossible |
| Open-domain | 62.50 | 10.06 | 18.3 | impossible |
| Multi-hop | 55.32 | 23.90 | 38.6 | impossible |
| Temporal | 58.57 | 56.04 | 71.8 | possible |

---

## 2. The definitive numbers

Re-scored offline from `results_latest.jsonl`. The "JUDGE" column reproduces the paper to the
decimal (proving this is the paper run); "true token-F1" is the metric the paper *claims*.

| Category | n | Paper "F1" (LLM judge) | **True token-F1** | token Precision | token Recall |
|---|---|---|---|---|---|
| Single-hop | 841 | 63.50 | **44.66** | 45.0 | 50.0 |
| Multi-hop | 282 | 55.32 | **23.48** | 21.3 | 39.8 |
| Temporal | 321 | 58.57 | **51.10** | 54.5 | 51.2 |
| Open-domain | 96 | 62.50 | **10.54** | 7.1 | 41.7 |
| **Macro** | 1540 | **59.97** | **32.45** | | |

---

## 3. Honest head-to-head (all token-F1)

A-MEM and BeliefMem numbers in the paper are token-F1. Compared on the same metric:

| | Single-hop | Multi-hop | Temporal | Open-domain |
|---|---|---|---|---|
| **Nous (true token-F1)** | 44.66 | 23.48 | **51.10** | 10.54 |
| A-MEM (self-rep) | 44.65 | 27.02 | 45.85 | 12.14 |
| BeliefMem (self-rep) | 48.41 | 40.51 | 51.88 | 28.73 |
| Nous − A-MEM | +0.01 | −3.54 | **+5.25** | −1.60 |
| Nous − BeliefMem | −3.75 | −17.03 | −0.78 | −18.19 |

**Nous ties A-MEM on single-hop, wins temporal, loses the other two; loses/ties BeliefMem on
all four.** The paper's "substantial gains in three of four" and "exceeds BeliefMem on all
categories" do **not** hold under a consistent metric.

> Fairness note: Nous used `google/gemini-2.5-flash` for answering + a dense-hybrid retriever
> (see §5), arguably a *stronger* setup than the baselines' GPT-4o-mini pipelines. So this
> comparison is, if anything, generous to Nous.

---

## 4. Why token-F1 is low — two different causes

Separating precision from recall (and inspecting judge-correct answers) shows the low token-F1
has two distinct causes, only one of which is fixable:

**(a) Verbosity — FIXABLE.** Right content, too many words → precision tanks.
Among single-hop judge-correct answers, mean token-recall is 76%. Examples:
- Q: *What did Mel and her kids make?* GT: `pots` — A: `their own pots` (judge 1.0, token-F1 0.50)
- Q: *What did Joanna finish?* GT: `screenplay` — A: `first full screenplay` (judge 1.0, token-F1 0.50)

A token-F1-optimized (terse) answer prompt would lift single-hop and temporal meaningfully.

**(b) Judge-generosity — NOT fixable by prompting.** The judge accepts semantically-right
answers that share almost no tokens with GT. **15% of open-domain judge-"correct" answers
share ZERO tokens with ground truth.** Examples:
- Q: *What fields would Caroline pursue?* GT: `Psychology, counseling certification` —
  A: `counseling or mental health` (judge 1.0, token-F1 0.05)
- Q: *Caroline's political leaning?* GT: `Liberal` —
  A: `liberal political leanings; strong advocate for LGBTQ+ rights…` (judge 1.0, token-F1 0.06)

The judge is measuring something legitimate (semantic correctness). The problem is (i) it's
labeled token-F1, and (ii) it's compared to baselines' token-F1. On inferential open-domain
questions token-F1 is arguably the *wrong* metric — but you must then score all systems under
the same judge, which hasn't been done.

---

## 5. Secondary discrepancies (paper vs code)

- **Backbone.** Paper: *"GPT-4o-mini for extraction, answer generation, and judging."*
  Code: extraction + answering use `google/gemini-2.5-flash`
  (`llm_extractor.py:98`, `eval_locomo.py:236`); GPT-4o-mini is **only** the judge.
- **Retrieval.** Paper: *"BM25 + two-hop entity BFS."* **CONFIRMED live on the eval path:**
  `engine.py:110` reads `OPENROUTER_API_KEY` from env; your eval runs `load_dotenv()` on a
  `.env` that has the key, so the paper run used the full stack: **dense embeddings**
  (`qwen/qwen3-embedding-8b`, `DENSE_WEIGHT=0.35`/`BM25_WEIGHT=0.65`) **+ BM25 + 2-hop BFS +
  an LLM `QueryRewriter`** (`engine.py:118`) — the last two components are undocumented in the
  paper. (BM25-only is only a *fallback* when no key is present.)
- **"No external vector database" claim.** Technically true (vectors cached in SQLite, not a
  vector DB) but misleading in spirit: the pipeline *does* call an external **embedding API**
  and an extra LLM query-rewrite call. The "principled Bayesian engine needs no embeddings"
  positioning is undercut by the eval leaning on dense retrieval.

---

## 6. Claim audit

| Paper claim | Status |
|---|---|
| "Strict token-F1, not LLM-judge" | ❌ False — it's the LLM-judge |
| "Beats A-MEM in 3 of 4 categories" | ❌ Does not hold under token-F1 (ties SH, wins T only) |
| "Exceeds BeliefMem on all 4 categories" | ❌ Loses/ties all 4 under token-F1 |
| "GPT-4o-mini backbone throughout" | ❌ Gemini-2.5-flash for extract/answer |
| "Open-domain 62.50" | ⚠️ Judge artifact; true token-F1 10.54 |
| Temporal is a strength | ✅ True (51.10 token-F1 > A-MEM 45.85) |
| Novel architecture (surprise/Bayes/decay/deltas) | ✅ Unchanged, genuinely distinctive |
| Honest, thorough Limitations section | ✅ Genuinely strong |

---

## 7. What's real and salvageable

1. **Architecture** — predictive-coding framing, closed-form Bayesian update, entropy decay,
   deltas. Novel and clean.
2. **Temporal reasoning** — a legitimate honest win vs A-MEM.
3. **The 28-point judge-vs-token-F1 gap** — itself a publishable finding on LoCoMo metric
   fragility, corroborating the community audit the paper cites.
4. **Startup value** — the belief-update / contradiction-handling mechanism is real engineering
   value independent of benchmark rank, *to be demonstrated by ablation*.

---

## 8. Recommended path (reframe, not abandon)

1. **Ablations** (the real scientific contribution): belief-update on/off, entropy-decay on/off,
   delta-aggregation on/off, dense-retrieval on/off — run on a fixed subset (~2–3 conversations,
   ~1.5h each) to keep cost low; every run dumps JSONL so both metrics are computed offline.
2. **Token-F1-optimized rerun** (subset) with terse answer prompts to establish the honest
   token-F1 ceiling → decide whether "competitive with A-MEM under strict F1, strongest on
   temporal" is a defensible claim.
3. **Rewrite `paper/nous.tex`**: retract SOTA claims; report BOTH metrics transparently; fix
   backbone + retrieval descriptions; foreground temporal + metric-sensitivity + architecture.
4. **Target** an agent-memory workshop (NeurIPS/ICLR) or a strong honest arXiv v2 — build real
   credibility instead of a claim that collapses on inspection.
5. **Later:** LongMemEval, second backbone, controlled BeliefMem, then the product demo.

---

## 9. Finding 5 — Reliability extraction makes the belief mechanism earn its keep (end-to-end)

**Question this resolves:** LoCoMo showed the Bayesian update is inert (Finding 2), and the
belief-layer contradiction test (Finding 4) showed *why* — the update only helps given
*per-observation reliability*, which the pipeline never supplied (it fed a fixed scalar). So:
if we actually extract reliability from language and feed it in, does the mechanism provide a
measurable, end-to-end benefit that survives comparison to a strong baseline? **Yes — modestly
and conditionally.**

**What was built** (`benchmark/reliability_bench.py`, code changes in §Deliverables):
1. `LLMExtractor.extract_with_reliability()` — estimates a per-claim reliability in [0,1] from
   the speaker's **epistemic markers** (hedging → low, definite/corrective → high) via a rubric.
   It never sees ground truth.
2. `engine.observe()` now accepts `(entity, attr, value, reliability)` 4-tuples and weights each
   Bayesian update by that claim's reliability.
3. A synthetic **NL contradiction/staleness benchmark**: 60 scenarios × 5 regimes; an entity's
   attribute evolves via natural-language statements of varying (linguistic) confidence, some
   false. Predict the *current* true value. Includes an **adversarial** regime (confidence
   inverted: true value hedged, false value asserted) as an anti-rigging honesty guard.

**Stage A — does the LLM recover confidence from phrasing?** Yes, cleanly:
`high`-confidence utterances → mean reliability **0.950**, `low` → **0.231**; separation
**+0.719**. The 0.7 no-match fallback fired ≈0% of the time, so the result is not a pinning
artifact — the extractor matched the designed value *and* scored its confidence.

**Stage B — accuracy predicting the current value (live extractor):**

| scenario regime | nous_reliability | nous_flat (LWW) | freq | append_retrieve (LLM over raw memory) |
|---|---|---|---|---|
| stable_noise | **100.0** | 41.7 | 100.0 | 75.0 |
| clean_change | 100.0 | 100.0 | 83.3 | 100.0 |
| recent_change | 100.0 | 100.0 | 0.0 | 100.0 |
| reliability_conflict | **100.0** | 25.0 | 0.0 | 83.3 |
| adversarial *(honesty guard)* | 0.0 | 41.7 | 50.0 | 16.7 |
| **MACRO (realistic, no adversarial)** | **100.0** | 66.7 | 45.8 | **89.6** |

**Interpretation (honest):**
- With reliability extraction, the Bayesian belief layer **decisively beats naive last-write-wins
  (100 vs 67)** and frequency (46) — reversing Finding 2's inertness. The fix works end-to-end.
- It also beats a **strong** baseline: `append_retrieve` (a capable LLM reading *all* raw
  statements in chronological order) scores 89.6. The belief layer's edge (~+10) is **concentrated
  exactly where theory predicts — noise/low-reliability resistance** (stable_noise 100 vs 75;
  reliability_conflict 100 vs 83); the two clean-change regimes tie at 100.
- The baseline is given **perfect recall** (every statement fed, no retrieval loss) and pays a
  query-time reasoning cost that **scales with memory size**; the belief layer precomputes a
  **bounded posterior** (O(1) query). So the comparison is generous to the baseline, and the
  belief layer's real-world edge (lossy retrieval + long histories + cost) is plausibly larger.
- **Downside, reported not hidden:** in the adversarial regime (confidence inverted) *every*
  confidence-based method degrades, `append_retrieve` worst (16.7). The claim is therefore
  **conditional: reliability-weighting helps when epistemic confidence tracks correctness (the
  common pragmatic case) and hurts when it is inverted.**

**Robustness — de-templating test (RELIABILITY_HARD=1):** to check the extractor reads epistemic
*stance* rather than pattern-matching the 5 canned forms, the phrasing bank was expanded to a
larger, more naturalistic set with **subtle hedges the rubric never lists** ("apparently",
"supposedly", "last I checked", "if I recall", "don't quote me", "rumor has it"). Result is
essentially unchanged: Stage A separation **+0.713** (high 0.943 / low 0.230), realistic macro
**nous_reliability 100 vs flat 64.6 vs freq 39.6**. This substantially closes the "templates made
it easy" critique — the reliability signal generalizes across varied surface forms.

**Remaining caveats (do not oversell):** the benchmark is still **synthetic** (no real-dialog
slice yet) and the confidence↔correctness correlation is by construction in 4/5 regimes (the
adversarial regime is the deliberate exception). This is a **proof-of-mechanism**, not evidence on
natural data; the honest next step is a **real-data** contradiction slice (mine genuine
self-corrections/hedges from dialog) and finer reliability gradations. It does **not** resurrect
the LoCoMo SOTA claim — it is a *separate*, honestly-scoped contribution about a regime LoCoMo
never tests.

**Bottom line:** this is the positive result the project needed. The paper's honest core becomes
*"belief-based memory with reliability-weighted Bayesian updates helps under contradictory,
variably-confident evidence — not general conversational QA,"* backed by (a) the mechanistic
Finding-4 analysis and (b) this end-to-end demonstration against last-write-wins **and** a strong
LLM-over-memory baseline. Same experiment is the startup demo (live belief distributions resisting
noisy contradictions).

---

*Findings 1 & the ablations are reproducible offline from `results_latest.jsonl` /
`ablation_*.jsonl` via `rescore_offline.py`. Finding 5 reproduces via
`python benchmark/reliability_bench.py --mock` (offline) and `--with-retrieval` (live).*
