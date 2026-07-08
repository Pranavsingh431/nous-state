# CLAUDE.md — Nous project working memory

This file is the durable record of the Nous audit + validation work. Read it first in any
new session. It is detailed on purpose.

---

## 0. What Nous is

**Nous** (arXiv:2606.22030, solo author Pranav Singh, IIT Ropar) is an agent-memory
architecture built on "knowledge is prediction, not storage." Instead of storing facts, it
keeps a **categorical probability distribution per entity-attribute pair** ("dimension"),
updates it with a **closed-form Bayesian posterior** scored by **information-theoretic
surprise** (`S = -log2 P`), stores each update as a **delta**, and models forgetting as
**entropy decay**. Identity resolution uses KL divergence between dimension sets.

**Goal (user's, dual):** (1) strong research → a top-tier / honest venue; (2) a startup/product.
The user has authorised changing the architecture if warranted.

### Repos / folders on disk
- `/Users/pranavsingh/Desktop/3.Current work/sarvam/` — **the real dev repo** (code, paper,
  benchmark, data, `.env`). All work below happens here.
- `/Users/pranavsingh/Desktop/3.Current work/Nous/` — session cwd; holds `Nous-preprint.pdf`
  and a fresh clone of the public `nous-state` GitHub repo (a cleaned subset of `sarvam`).
- `.env` in `sarvam/` contains `OPENROUTER_API_KEY` (the user's rotated key). The eval reads it.

### Architecture code map (`sarvam/nous/`)
`dimension.py` (categorical dist + entropy decay), `surprise.py` (`-log2 p`, KL, MI),
`updater.py` (Bayesian posterior), `delta.py`/`delta_log.py`, `world_model.py`,
`engine.py` (1.2k lines — orchestration, ingest `observe()`, query `query_relevant()`,
`_get_observed_values` aggregation, hybrid retrieval init), `retrieval.py` (BM25 + dense +
graph BFS), `llm_extractor.py` (LLM triple extraction), `config.py`, `persistence.py` (SQLite).

---

## 1. THE THREE MAJOR FINDINGS (most important section)

### Finding 1 — The paper's "F1" is a generous LLM judge, not token-F1
- Paper claims **strict token-level F1** and uses that to justify not comparing to Zep/Mem0.
- Actual eval (`benchmark/eval_locomo.py:585`) aggregates `llm_judge_score()` — a GPT-4o-mini
  judge prompted **"Be GENEROUS"** — under a variable named `f1`. Real token-F1 (`f1_score`)
  is only an exception fallback. True at the paper commit `5c8ffcc`.
- **Proven from the paper's own numbers:** token-F1 ≤ `2·BLEU1/(BLEU1+1)`. 3 of 4 reported
  "F1" values exceed that ceiling → they cannot be token-F1. Confirmed by re-scoring
  `results_latest.jsonl` (reproduces paper's 63.50/55.32/58.57/62.50 exactly as the JUDGE col).

**Honest numbers (full 1,540-question paper run, re-scored offline):**

| Category | Paper "F1" (=judge) | TRUE token-F1 |
|---|---|---|
| Single-hop | 63.50 | 44.66 |
| Multi-hop | 55.32 | 23.48 |
| Temporal | 58.57 | 51.10 |
| Open-domain | 62.50 | 10.54 |
| **Macro** | **59.97** | **32.45** |

Under a consistent metric, Nous **ties A-MEM on single-hop, wins temporal, loses multi-hop +
open-domain, and loses to BeliefMem on all four.** The SOTA claim does not hold.

Two more paper-vs-code gaps: **backbone** is `google/gemini-2.5-flash` for extract+answer (not
GPT-4o-mini; that's only the judge); **retrieval** was the full dense+BM25+BFS+query-rewrite
stack (paper says "BM25 + 2-hop BFS"). Details in `RESULTS_AUDIT.md`.

### Finding 2 — The Bayesian belief mechanism is INERT on LoCoMo (the big one)
Controlled ablation, 3-conversation subset (385 questions), fast parallel eval, BM25+BFS:

| Variant | JUDGE macro | token-F1 macro |
|---|---|---|
| `bm25_full` (full engine) | 69.62 | 36.77 |
| `bm25_flat` (Bayesian update → naive last-write-wins overwrite) | **72.46** | 36.27 |
| `bm25_noagg` (observed-values aggregation off) | 66.58 | 35.39 |

**Turning off the Bayesian update does not hurt — it's slightly better on judge, a wash on
token-F1.** The paper's central intellectual contribution (predictive world model / Bayesian
belief tracking) provides **no measurable benefit** on LoCoMo. Contrast BeliefMem, whose
analogous ablation dropped F1 42→20.

Interpretation: most LoCoMo questions are about facts that don't oscillate, so "keep the latest
value" (last-write-wins) ties the full Bayesian machinery. LoCoMo does not stress the thing
Nous is designed for (contradictory / noisy / uncertain evidence).

### Finding 3 — What actually works: aggregation (a little), not the belief math; dense is inert
- **Observed-values aggregation helps**, mainly multi-hop: `no_agg` drops multi-hop judge
  55.41 → 50.00 (−5.4), macro judge −3.0. But this is a **delta-log listing trick** ("list all
  values ever seen"), which works precisely by *ignoring* the unimodal posterior — orthogonal
  to the Bayesian thesis.
- **Dense retriever is inert.** conv-1 fair comparison: dense judge 63.34 vs BM25 65.26; token-F1
  35.76 vs 34.79. BM25+BFS (the paper's *described* method) is as good, and dense is slow/flaky.
- **Entropy decay was never called during the eval** — the benchmark says nothing about it.

---

### Finding 4 — Bayesian helps ONLY with reliability signals (the key mechanistic result)
Contradiction micro-benchmark (`benchmark/contradiction_bench.py`, belief-layer only, no LLM,
400 sequences/cell). Compares `nous_bayes` vs `flat_lww` (last-write-wins) vs `freq` (most
frequent) at predicting the current true value under evolving/contradictory evidence.

| scenario | constant reliability (bayes/flat/freq) | varying reliability (bayes/flat/freq) |
|---|---|---|
| stable_noise | 35.0 / 34.8 / **100** | **100** / 37.5 / 100 |
| clean_change | 100 / 100 / 60 | 100 / 100 / 61 |
| recent_change | 100 / 100 / 0 | 100 / 100 / 0 |
| noisy_change | 69.5 / **100** / 78 | 100 / 100 / 80 |
| reliability | 31.0 / 32.8 / 0 | **100** / 34.2 / 0 |
| **MACRO** | **67.1 / 73.5 / 47.7** | **100 / 74.3 / 48.2** |

- **Constant reliability = the real Nous pipeline** (`observe()` feeds a fixed reliability; the
  extractor produces none). Here the Bayesian update **degenerates to a recency-follower and is
  slightly WORSE than last-write-wins** (67.1 vs 73.5). This mechanistically explains Finding 2
  (LoCoMo inertness): with constant reliability, the posterior ≈ soft last-write-wins.
- **Varying reliability = the mechanism's ceiling.** Given per-observation reliability, Bayesian
  is perfect (100 macro) and dominates flat (74) and freq (48), winning exactly where theory
  predicts (noise resistance, trustworthy-but-infrequent sources).
- **The missing architectural piece is reliability extraction.** The Bayesian engine is correct
  but starved of the one input that makes it worthwhile. Adding a reliability-extraction layer is
  the evidence-motivated fix (the user authorised architecture changes).
- Caveat: absent reliability, plain frequency-counting beats Bayesian on noise resistance
  (100 vs 35) — the posterior isn't even the best simple option without reliability.

### Finding 5 — Reliability extraction makes the mechanism earn its keep END-TO-END (the fix works)
The Finding-4 fix was built and tested through the real pipeline. New code: `extract_with_reliability()`
(per-claim reliability from epistemic markers, never sees ground truth) + `observe()` accepts
`(entity,attr,value,reliability)` 4-tuples and weights each Bayesian update by it. New benchmark
`benchmark/reliability_bench.py`: 60 NL scenarios × 5 regimes; facts evolve via statements of
varying *linguistic* confidence (some false); predict the current value. Includes an **adversarial**
inverted-confidence regime as an anti-rigging guard.

- **Stage A (does the LLM recover confidence from phrasing?):** yes — `high`→mean **0.950**,
  `low`→**0.231**, separation **+0.719**; the 0.7 no-match fallback fired ≈0% (not a pinning artifact).
- **Stage B (live extractor), accuracy predicting current value:**

| regime | nous_reliability | nous_flat (LWW) | freq | append_retrieve (LLM over raw memory) |
|---|---|---|---|---|
| stable_noise | **100** | 41.7 | 100 | 75.0 |
| clean_change | 100 | 100 | 83.3 | 100 |
| recent_change | 100 | 100 | 0 | 100 |
| reliability_conflict | **100** | 25.0 | 0 | 83.3 |
| adversarial (guard) | 0 | 41.7 | 50 | 16.7 |
| **MACRO (realistic, no adversarial)** | **100** | 66.7 | 45.8 | **89.6** |

- **Reverses Finding 2's inertness:** with reliability, Bayesian **beats last-write-wins 100 vs 67**,
  and beats even a *strong* baseline — `append_retrieve` (capable LLM reading ALL raw statements) 89.6.
  The ~+10 edge is **concentrated in noise resistance** (stable_noise 100 vs 75; reliability_conflict
  100 vs 83); clean/recent change tie at 100.
- The baseline gets **perfect recall** (no retrieval loss) and pays a query cost that **scales with
  memory**; the belief layer precomputes a **bounded O(1)** posterior — so the comparison is generous
  to the baseline and the real-world edge is plausibly larger.
- **Honest downside (reported):** adversarial/inverted confidence breaks *all* confidence-based methods
  (append_retrieve worst, 16.7). Claim is **conditional: helps when confidence tracks correctness
  (common case), hurts when inverted.**
- **Robustness ✅ (de-templating done):** `RELIABILITY_HARD=1` swaps in a larger, naturalistic
  phrasing bank with subtle hedges the rubric never lists ("supposedly", "apparently", "last I
  checked", "don't quote me"...). Result essentially unchanged — Stage A sep **+0.713**, realistic
  macro **100 / 64.6 / 39.6**. So the extractor reads epistemic *stance*, not template patterns;
  the "templated" critique is substantially closed.
- **Remaining caveats — do NOT oversell:** still **synthetic** (no real-dialog slice yet); the
  confidence↔correctness correlation is by construction in 4/5 regimes (adversarial is the
  exception). Proof-of-mechanism, not real-data evidence; does **not** resurrect the LoCoMo SOTA
  claim. Next: a **real-data** contradiction slice + finer reliability gradations. Full writeup:
  `RESULTS_AUDIT.md` §9.

### Finding 6 — Poisoning resistance is REAL but gated entirely by provenance trust (the fork result)
The falsifiable experiment (`benchmark/poison_bench.py`, belief layer, offline, 400 trials/cell).
Establish a true belief with 4 trusted obs (r=0.9); attacker injects M false obs through a channel
of trust r_poison; ASR = % of attacks that flip the belief. Strategies: `nous_trust` (reliability=
source trust), `nous_notrust` (poison wrongly trusted at 0.9 — ablation), `lww`, `majority`.

| channel trust | nous_trust ASR (M=1→50) | majority | lww | nous_notrust |
|---|---|---|---|---|
| r=0.2 (untrusted) | **0% at every M incl. 50:4** | 0%→100% (M≥5) | 100% | 100% |
| r=0.5 (mid) | **0% at every M** | same | 100% | 100% |
| r=0.9 (forged/trusted) | **100%** (fails) | — | 100% | 100% |

Crossover (clean, not a 2-value artifact — identical at k=4; k=8 similar): **ASR=0% for r_poison ≤0.5
at ALL volumes**; r=0.55 holds to ~M=10 then flips; r≥0.6 flips at M=3; r≥0.7 flips at M=1.

**What it means (honest):**
- Real, sharp security property: source-trust-weighted belief updating gives **0% poisoning ASR when
  the attacker is confined to a channel trusted below ~0.5, even under a 50:4 volumetric flood**,
  where append/majority/last-write-wins all hit 100%. `nous_notrust`=100% proves the **source-trust
  signal, not the Bayesian math**, is the defense.
- **Load-bearing assumption, stated sharply:** the defense holds *iff trust is assigned by PROVENANCE
  the attacker can't forge* (which channel it came from), NOT by content they write. If trust is
  inferred from content/phrasing (the epistemic-marker extractor), a confident-sounding attacker
  earns r≈0.9 → ASR 100% (the forged-trust column). So the product is **NOT "AI that detects poison"**
  — it's **trust-partitioned memory**: tag every source with a provenance trust tier; poison from an
  untrusted tier cannot move a belief. Poisoning a Nous agent requires compromising a *trusted source*.
- Maps directly to OWASP ASI06's named defense direction (trust scoring + provenance + trust-aware
  retrieval) — a problem the security community says nobody does well yet.

**Limits of THIS test (do NOT overclaim):** belief-layer only (no retrieval/LLM); trust was given as
an oracle (real question: can provenance-trust be assigned robustly in practice?); the "r<0.5 poison
= evidence FOR truth" behavior is an aggressive modeling assumption. **Next diagnostic:** full-pipeline
poisoning mapped to AgentPoison/MINJA, testing whether an attacker can game *content-inferred* trust
(which the forged-trust column predicts breaks it) vs. provenance trust (which holds).

### Finding 7 — Full-pipeline poisoning: content-trust is gameable (measured); provenance-capped trust holds
`benchmark/poison_pipeline_bench.py` — the fork-resolving experiment, with the REAL Finding-5
extractor reading adversarial text. Threat: attacker on an untrusted channel (tier 0.2) injects
confident-sounding false claims ("Confirmed: database is MongoDB…") to flip a trusted belief
(Postgres, tier 0.9). Tests whether an attacker can LAUNDER low-provenance content into high trust.

- **Stage 1 (real extractor on adversarial text):** confident poison → mean reliability **0.96**
  (one 1.0); hedged poison → **0.25**. So **content-inferred trust (Finding 5) is adversarially
  GAMEABLE — measured, not assumed**: a confident attacker earns r≈0.96 from the extractor alone.
- **Stage 2 (Attack Success Rate under 4 trust models):**

| trust model | M=1 | M=3 | M=10 | meaning |
|---|---|---|---|---|
| `content_only` (Finding-5 phrasing trust) | **100%** | 100% | 100% | broken by ONE confident injection |
| `provenance_only` (channel tier) | **0%** | 0% | 0% | holds |
| `hybrid_min` = min(content, provenance) | **0%** | 0% | 0% | **the correct design** |
| `hybrid_max` = max(content, provenance) | **100%** | 100% | 100% | "boost authoritative content" = the hole |

**Resolution (reconciles Findings 5 & 6):** the security property survives falsification, but ONLY
under **provenance-capped trust**: `trust = min(provenance_tier, content_confidence)` — content may
only ever *lower* trust within a provenance ceiling, never *raise* it. Content-only trust and any
content-boosting (max) design are broken by a single confident injection. Architectural rule for
the product/paper: **never let content raise trust above its channel's provenance tier.**

**Honest residual limits (the next real unknowns — do NOT skip when pitching/writing):**
1. Belief layer only — not a real retrieval/reasoning attack. Need provenance-AWARE retrieval too.
2. `provenance_only` holding is partly by construction. The real threat it does NOT cover:
   **provenance laundering / indirect injection** — attacker content entering via a *trusted*
   intermediary (a trusted tool that ingests attacker-controlled data) inherits a trusted tier →
   back to 100%. This is the genuine next diagnostic.
3. The `min` cap has an unmeasured **benign utility cost**: a genuinely reliable statement on a
   low-trust channel is under-weighted (Finding-5 benefits shrink there). Security/utility tradeoff.
4. Threat modeled as AgentPoison/MINJA *pattern*, NOT a replication — verify exact specs vs primary
   papers before claiming numbers "comparable to published work."

### Finding 8 — The hard problem is not the update rule; it's taint-tracked provenance (+ its utility cost)
Two belief-layer experiments (offline, 400 trials/cell) resolving the Finding-7 fork.

**8a. Provenance laundering** (`benchmark/poison_launder_bench.py`) — confident poison enters via a
TRUSTED intermediary. Trust = min(provenance, content); attacker content_conf≈0.96.

| policy | eff. r | ASR (M=1→50) |
|---|---|---|
| `direct` (attacker on untrusted channel) | 0.20 | **0%** (Finding-7 control) |
| `naive_inherit` (laundered poison takes intermediary's tier 0.9) | 0.90 | **100%** at every volume |
| `taint_propagate` (down-tier to untrusted origin 0.2) | 0.20 | **0%** |

→ **Laundering FULLY defeats provenance-capped trust without taint-tracking (100%). It holds (0%)
only if provenance propagates through every trusted intermediary.** The security claim is real but
now conditional on a *taint-tracking* provenance system — a hard, known-hard requirement (information-
flow control). The belief update was never the hard part; **provenance assignment + propagation is.**

**8b. min-cap utility cost** (`benchmark/mincap_utility_bench.py`) — accuracy when the reliable-but-
infrequent TRUE source is routed through a low provenance tier and capped.

| regime | content_only | t_true=0.9 | 0.7 | 0.5 | 0.3 | 0.2 |
|---|---|---|---|---|---|---|
| reliability_conflict | 100% | 100 | 100 | 100 | 88.0 | 53.8 |
| stable_noise | 100% | 100 | 100 | 100 | 21.2 | 8.8 |

→ **The cap is FREE when genuine sources sit at tier ≥0.5, but destroys benign accuracy when reliable
info is stuck at ≤0.3** (down to 9–54%). So tier-assignment quality is load-bearing for utility too.

**Synthesis (both point to the same crux):** the entire security-AND-utility profile is governed by
one thing outside the belief engine — **the quality of provenance-tier assignment + taint propagation.**
The belief update (min-capped Bayesian) is sound and its boundaries are now precisely mapped. What is
unproven and hard is the provenance layer. Implication for paper vs product:
- **Paper: ready now.** Honest, novel, workshop-tier — "provenance-capped belief updating defends
  against memory poisoning iff provenance is taint-tracked; we map the crossover and the utility cost."
  Matches the user's PRIMARY goal (publication). The belief engine + Findings 5–8 are the paper.
- **Product: further out, riskier.** The belief engine is necessary but NOT sufficient; the sellable
  defense requires the taint-tracking provenance stack, which is the actual hard problem and a bigger
  build. Viable ONLY where the host agent already exposes provenance metadata (web-fetch vs tool vs
  user); opaque trusted intermediaries (a RAG connector pulling a poisoned doc) are the failure mode.
  Do not commit to the product until a provenance-integration prototype is tested against laundering.

### Finding 9 — End-to-end agent test: offline findings REPRODUCE in a real pipeline; tradeoff is concrete
`benchmark/provenance_agent_bench.py` — minimal provenance-aware coding assistant over the real
sarvam/nous repo. Two memory backends (append-retrieve baseline vs min-capped nous belief layer),
IDENTICAL LLM answerer + prompt, only memory differs. Real extractor for content-confidence, real
LLM for answers. Entity/attr/value pinned (true & poison compete in one belief; isolates trust var).

- **Utility: 80% vs 80% — but the aggregate is a DESIGNED tie, not a discovery.** The 5-scenario set
  was balanced on purpose. The real result is the symmetric pair: **U3 (legit low-tier web correction)
  → nous WRONG (kept stale high-tier value), baseline right; U4 (malicious low-tier contradiction) →
  baseline WRONG (flipped), nous right.** Same mechanism, opposite ground truth. Nous trades benign
  low-tier responsiveness for poison resistance. Adding more U3-type items would drop nous below
  baseline; the 80/80 is not evidence of "no utility cost" — U3 IS the cost, made concrete.
- **Direct poison (web channel): baseline 100% ASR at 1/3/10 injections; nous 0%.** Findings 6/7
  survive the real pipeline end-to-end.
- **Laundering (poison via trusted tool): baseline 100%, nous-naive 100%, nous-taint 0%.** Finding 8a
  reproduced end-to-end: naive tiering defeated; taint-tracking mandatory.

**Precise claim supported:** nous is BETTER when false info arrives on a correctly-low-tiered channel
(direct poison: 0% vs 100%), WORSE when correct info arrives on a low-tier channel (U3), and under
laundering only as good as its taint-tracking. The trade is governed entirely by provenance quality
(correct tiers + taint propagation), not the belief math — consistent with Findings 6–8.

**Limits (honest):** tiny hand-built scenario set (aggregate numbers are illustrative, not
statistical); pinned entity/attr/value means extraction/entity-linking fragility (a real nous cost in
the wild) is NOT tested; temp=0 single-trial (clean 0/100 ASR, no variance); nous's poison "answer"
is really its pre-resolved belief surfaced through the LLM (expected, not independent reasoning).
Load-bearing tier assignment is auditable in one place: `CHANNEL_TIER` + `provenance_tier()`.

### Finding 10 — Extraction fragility: a real UTILITY tax (~33 pts), but security is unharmed
`benchmark/provenance_unpinned_bench.py` + probes. Dropped Finding 9's pinning so the REAL extractor
decides entity/attr/value; measured the cost end-to-end.

- **Extraction is messy: 3/3 poison facts FRAGMENTED** — true & poison filed under different
  entities/attributes (e.g. true→`Nous Library.data_persistence=SQLite`, poison→`Nous.database=MongoDB`;
  db_path→`Db Path.default` vs `Db Path.status`). Non-deterministic: `default_model`'s true value was
  filed under `identity` in one run and dropped entirely in another.
- **Security UNHARMED by fragmentation.** Direct-poison ASR stayed **0% unpinned** (baseline 100%),
  and laundering still needs taint (naive 100% / taint 0%). Mechanism: low-trust poison (r=0.2) never
  even becomes the argmax of its own fresh dimension — it stays `unknown` at any volume (r≤0.5 property,
  Finding 6). So fragmentation doesn't help the attacker.
- **Utility TAXED by fragmentation — and my first utility test hid it.** The harness's aggregate said
  "unpinned 80% == pinned 80%, extraction cost +0" — MISLEADING (the 5 scenarios extracted favorably).
  A clean probe (3 repo facts, true-only, no poison) gives the real figure: **pinned 100% → unpinned
  66%, a ~33-pt extraction tax** (`default_model` became unanswerable — agent said `unknown` despite
  holding the fact under the wrong attribute, or dropped it). Tiny n=3, so illustrative not statistical.

**Corrected honest tradeoff:** nous's SECURITY (poison resistance) is robust even under noisy
extraction, but its UTILITY carries an extraction tax that append-and-retrieve does NOT pay (baseline
stores raw text, never misfiles/loses a fact). So nous wins where poison-resistance matters AND facts
are cleanly/structurally extractable; it loses to plain retrieval where extraction is unreliable and
you just need raw recall. This is a real cost of the structured-belief approach, now quantified.

### Finding 10b — Extraction tax at scale is SMALL (~3%), not 33%; the n=3 scare was noise
`benchmark/extraction_tax_bench.py` — 30 real repo facts (config defaults, signatures, README),
true-only, pinned vs unpinned, identical render-all answerer. **Tax = 3 pts (pinned 100% → unpinned
97%, n=30).** Only ONE genuine extraction drop (`max_tokens` → extractor returned nothing). The n=3
Finding-10 "33%" was small-sample noise + a MATCHER false-negative (my own harness again: it scored
`0.8`≠`0.80` and `zero`≠`none` as wrong; fixed via numeric-equivalence matching before the final run).
**Caveats:** this is a FLOOR — render-all is generous (misfiled beliefs still visible to the LLM);
real retrieval would miss facts filed under the wrong entity/attr and raise the tax. Single run, mild
LLM non-determinism, favorable short-valued technical domain. **Implication:** extraction is NOT the
product-killer Finding 10 implied — it removes the biggest apparent objection. The real remaining
costs are the U3 low-tier-responsiveness tradeoff and the taint-tracking requirement (both mapped).
**Experimental program (Findings 1–10) is now COMPLETE — freeze belief-layer research; next is
writing, not measuring.**

### Finding 11 — External verification (web, Jul 2026): threat specs real, A-MEM numbers were WRONG, defense space occupied
Verified the two pre-writing items the plan flagged. Results materially change positioning.

**11a. Threat-model specs (verified against primary papers):**
- **AgentPoison** (arXiv:2407.12784, NeurIPS 2024): optimized-trigger BACKDOOR in memory/RAG; ASR ≥80%,
  poison rate <0.1%, ≤1% benign impact. Autonomous-driving/QA/EHR agents.
- **MINJA** (arXiv:2503.03704, NeurIPS 2025): QUERY-ONLY injection via bridging steps; 98.2% injection,
  76.8% ASR. → **Our threat model (volumetric fact-overwrite + provenance defense) is NOT a replication
  of either** (they are trigger-backdoor / reasoning-bridging attacks). MUST NOT claim comparable ASR.
  Cite as establishing the threat CATEGORY only.
- **OWASP ASI06 "Memory & Context Poisoning"** CONFIRMED in the 2026 Agentic Top 10; its named
  mitigations EXPLICITLY include **"track data provenance," "provenance metadata on every memory write,"
  tenancy separation, expiry. → validates the provenance direction, but means provenance is a NAMED
  mitigation, not our invention. Our contribution = a concrete MECHANISM + empirical characterization.

**11b. Prior-work / novelty (IMPORTANT):** arXiv:2601.05504 (Jan 2026) "Memory Poisoning Attack and
Defense" already does trust-based defense: composite **content/pattern-based trust scoring** + trust-
aware retrieval + temporal decay; ASR→6.67% (GPT-4o-mini). Does NOT use source provenance or Bayesian
updating. → **Cannot claim "first trust-based poisoning defense."** Differentiation is: PROVENANCE
(not content) trust + our measured proof that content-inferred trust is adversarially gameable
(Finding 7) + the min-cap composition + the taint-tracking requirement. Honest and still novel.

**11c. A-MEM numbers — the paper's baseline is WRONG (a real integrity fix).** Primary A-MEM table
(arXiv:2502.12110, Table 1, GPT-4o-mini token-F1): **Single-hop 27.02, Multi-hop 45.85, Temporal 12.14,
Open-domain 44.65** (Adversarial 50.03). The Nous paper cites A-MEM as 39.41/32.86/31.23/17.10 — matches
the primary NOWHERE. The paper's "No memory" row {40.36,25.02,18.41,12.04} is A-MEM's LoCoMo baseline
ROTATED into wrong categories; **CLAUDE.md's Finding-1 A-MEM row had the SAME rotation.** Corrected
head-to-head, both token-F1 (backbone differs — see caveat):

| Category | Nous (true token-F1) | A-MEM (primary) | winner |
|---|---|---|---|
| Single-hop | 44.66 | 27.02 | **Nous +17.6** |
| Multi-hop | 23.48 | 45.85 | A-MEM +22.4 |
| Temporal | 51.10 | 12.14 | **Nous +39.0** |
| Open-domain | 10.54 | 44.65 | A-MEM +34.1 |

→ Honest record: **Nous wins Single-hop and Temporal DECISIVELY, loses Multi-hop and Open-domain
decisively (2–2)** — different from both the paper's "wins all 4" AND CLAUDE.md's earlier "ties SH,
wins T" (that was the rotation error). Temporal win (51 vs 12) is genuinely large and real.
**CAVEAT (unresolved):** Nous numbers are Gemini-2.5-flash (extract+answer); A-MEM is GPT-4o-mini.
Still not same-backbone. Clean comparison needs a Nous GPT-4o-mini rerun (~8h) OR an explicit caveat.

## 2. Current honest assessment

- The **architecture is clean and the framing is novel**, but on the benchmark used, the novel
  mechanism does not drive results — mundane retrieval + aggregation + category prompts do.
- The paper as written **cannot go to a top venue**: the headline metric is mislabeled, the
  SOTA claim inverts under a consistent metric, and the core mechanism is unvalidated (in fact
  shown inert) by its own ablation.
- This forced an honest pivot: test Nous's value proposition (contradiction/uncertainty handling)
  where it actually applies — which LoCoMo is not. **That make-or-break test has now been run
  (Finding 5) and the mechanism PASSED, conditionally:** given reliability extraction, the Bayesian
  belief layer beats last-write-wins (100 vs 67) and a strong LLM-over-memory baseline (100 vs 90)
  on contradictory, variably-confident evidence — but only when confidence tracks correctness, and
  so far only on a synthetic/templated benchmark. So the honest positioning is no longer "SOTA on
  LoCoMo" but **"belief-updating with reliability extraction helps under contradictory/uncertain
  evidence — a regime standard conversational QA doesn't test."** That is a real, defensible,
  novel contribution; it now needs a less-templated / real-data test to harden.
- Real progress: in two sessions we went from an unverified SOTA claim → a precise honest account
  of what doesn't work (Findings 1–3) → the mechanistic reason (Finding 4) → a built-and-validated
  fix with an end-to-end positive result (Finding 5). That is exactly the arc needed before
  submitting or pitching anything.

---

## 3. Deliverables created this session (all in `sarvam/`)

| File | Purpose |
|---|---|
| `RESULTS_AUDIT.md` | Full audit of Finding 1 (metric), with proofs, examples, claim table. |
| `benchmark/rescore_offline.py` | Re-scores any results JSONL → JUDGE vs TRUE token-F1. $0, no API. |
| `benchmark/eval_locomo_fast.py` | **Fast** parallel eval (16-way extraction + 8-way answer/judge). ~10× faster; behaviour-identical (updates stay ordered). |
| `benchmark/run_ablations.sh` | Runs the ablation suite via the fast eval; rescopes each on both metrics. |
| `benchmark/ABLATIONS.md` | What each ablation tests + how to read it. |
| `benchmark/ablation_*.jsonl` | Raw per-question results: `bm25_full`, `bm25_flat`, `bm25_noagg` (3 conv), `dense_full` (conv 1). |
| `benchmark/ablation_run.log` | Full ablation run log with per-variant timing + rescores. |
| `benchmark/contradiction_bench.py` | Finding 4: belief-layer contradiction micro-benchmark (offline, no LLM). |
| `benchmark/reliability_bench.py` | **Finding 5:** end-to-end reliability benchmark. `--mock` (offline) / live / `--with-retrieval`. |
| `benchmark/reliability_bench_{mock,live}.json` | Raw Finding-5 results + Stage-A reliability data. |
| `benchmark/reliability_live*.log` | Finding-5 live run logs (core + with-retrieval). |
| `paper/nous.tex` | **REWRITTEN around the honest arc** (Findings 1–5). Compiles → `nous.pdf`, 9pp, 0 errors. |
| `CLAUDE.md` | This file. |

### Paper rewrite (done — `paper/nous.tex`)
Retracted the SOTA claim everywhere (abstract, intro contributions, results, conclusion). Now:
honest token-F1 main table (Nous wins SH+T, loses MH+OD vs A-MEM); a **metric-sensitivity**
table exposing the 27.5-pt judge-vs-token-F1 gap; a new **Ablations** section (belief update inert,
aggregation helps, dense inert); a new **Reliability Extraction** section (Findings 4–5, both
tables); corrected backbone (gemini-2.5-flash) + retrieval; rewritten Limitations (ablations now
done; synthetic-benchmark + real-dialogue caveats) and Conclusion. A-MEM numbers used are the
paper's own cited set (39.41/32.86/31.23/17.10); the alt set in Finding 1 (44.65…) is unverified —
**flag: reconcile A-MEM's true token-F1 against the A-MEM paper before submission.** BeliefMem is
NOT actually cited in the tex (earlier "two pages on BeliefMem" was wrong) — add its citation if used.
Preamble fixes for a broken local TeX install: dropped `times` + `dblfloatfix`, added tikz
`positioning`; **4 corrupt user-local `.sty` files (404-HTML) moved to `~/Library/texmf/_corrupt_backup/`.**

### Code changes
Ablation toggles (default-OFF; production behaviour unchanged):
- `nous/config.py` — `ABLATE_UPDATE` / `ABLATE_AGG` / `ABLATE_DENSE` env flags.
- `nous/updater.py` — `NOUS_ABLATE_UPDATE=flat` → last-write-wins overwrite instead of posterior.
- `nous/engine.py` — `NOUS_ABLATE_DENSE=1` gates dense retriever; `NOUS_ABLATE_AGG=1` in
  `_get_observed_values`; **`observe()` gained an optional `claims=` param** so extraction can be
  parallelized while belief updates stay strictly ordered.

Reliability extraction (Finding 5; additive, backward-compatible):
- `nous/llm_extractor.py` — new `extract_with_reliability()` (returns 4-tuples with per-claim
  reliability from an epistemic-marker rubric) + `_build_reliability_prompt()` + shared `_chat()`
  helper. Battle-tested `extract()` **untouched** (still on the LoCoMo eval path).
- `nous/engine.py` — `observe()` now accepts 3- OR 4-tuple claims; a 4th element overrides the
  call-level `reliability` per claim (threaded into `updater.update` and relationship edges).

---

## 4. How to reproduce (all commands from `sarvam/`)

```bash
# Honest re-score of the paper run (instant, no API):
python benchmark/rescore_offline.py benchmark/results_latest.jsonl

# Fast eval, 1 conversation, BM25-only (validation, ~a few min):
NOUS_ABLATE_DENSE=1 MAX_CONVERSATIONS=1 OUTPUT_JSONL=benchmark/fast_smoke.jsonl \
  python -u benchmark/eval_locomo_fast.py

# Full ablation suite (3 conv, 4 variants; BM25 variants ~7-8 min each):
MAX_CONVERSATIONS=3 bash benchmark/run_ablations.sh

# Re-score any variant offline (both metrics):
python benchmark/rescore_offline.py benchmark/ablation_bm25_flat.jsonl

# Finding 5 — reliability benchmark:
python benchmark/reliability_bench.py --mock            # offline, perfect reliability, no API
python benchmark/reliability_bench.py                   # live extractor (reads .env key)
python benchmark/reliability_bench.py --with-retrieval  # + append_retrieve LLM baseline
```

Notes: the fast eval defaults to `INGEST_WORKERS=16`, `QA_WORKERS=8`. Dense endpoint
(`qwen3-embedding-8b`) is flaky/slow via OpenRouter — keep dense OFF for routine runs.
BM25+BFS ≈ dense on quality and matches the paper's described method.

---

## 5. Plan / next steps

### ⛔ MEASUREMENT IS CLOSED. Findings 1–11 close the loop. The next action is WRITING, not experiments.
Every experiment that needed running has run (mechanism, ablations, contradiction, reliability,
poisoning, laundering, utility cost, extraction tax, external verification). **No further experiments
are authorized.** If a fresh session (or a "quick necessary check") proposes a new run, that is the
scope-creep failure mode we explicitly named — decline it. The one open confound (A-MEM backbone
mismatch, Gemini vs GPT-4o-mini) is a **one-sentence Limitations note, NOT a rerun.** The GPT-4o-mini
subset diagnostic was deliberately killed mid-flight to stop the drift; do not restart it.

### THREE HONESTY CONSTRAINTS the paper MUST respect (they silently revert to overclaiming if unlocked):
1. **A-MEM is 2–2, not a sweep.** Nous WINS single-hop (44.66 vs 27.02) and temporal (51.10 vs 12.14,
   a large real win), LOSES multi-hop (23.48 vs 45.85) and open-domain (10.54 vs 44.65). Same-metric
   token-F1; backbone differs (→ Limitations sentence). Never write "beats A-MEM on all four."
2. **Novelty is narrow.** Do NOT claim "first trust-based poisoning defense" — arXiv:2601.05504 (Jan
   2026, content/pattern trust scoring, ASR→6.67%) got there. The four real contributions: (a)
   provenance-not-content trust, (b) measured proof content-trust is adversarially gameable (Finding 7),
   (c) the `min(provenance, content)` composition rule, (d) the taint-tracking requirement (Finding 8a).
   ✅ PRIMARY-VERIFIED (arxiv.org/abs/2601.05504): its defense is "composite trust scoring across
   multiple orthogonal signals" + "memory sanitization with trust-aware retrieval, temporal decay,
   pattern-based filtering" — i.e. CONTENT/PATTERN trust, NO provenance, NO Bayesian, NO "first" claim.
   → Their content-based trust is exactly what our Finding 7 shows is gameable; cite as the closest
   prior trust-defense and position provenance-capped Bayesian updating as the distinct contribution.
3. **Threat models are a CATEGORY reference, not a benchmark comparison.** Cite AgentPoison (≥80% ASR)
   and MINJA (76.8% ASR) to establish memory poisoning is a real studied threat. Do NOT claim our ASR
   numbers are comparable — different attack shape (trigger-backdoor / bridging vs fact-overwrite).

### PAPER STATUS (as of this session): plan APPROVED (one paper, whole arc; §4/§5 first, then framing).
nous.tex (822 lines) was already partly rewritten for Findings 1–5 (honest metric framing, backbone,
ablations §, reliability §) — but had the WRONG phantom A-MEM numbers (39.41/32.86/31.23/17.10).
- ✅ **A-MEM integrity fix APPLIED** this session: Table 1 + abstract + §2 + §5 prose now use the
  PRIMARY numbers (SH 27.02, MH 45.85, T 12.14, OD 44.65; MemGPT 26.65/25.52/9.15/41.04; LoCoMo row
  25.02/18.41/12.04/40.36). Honest record stated as **2–2** (Nous wins SH+T, A-MEM wins MH+OD).
  Verified no phantom numbers remain (grep clean). Cross-check: primary corroborated by rotation-match.
- ✅ **2601.05504 primary-verified** (hard gate satisfied): content/pattern trust, no provenance/Bayes.
- ⏳ **STILL TODO (task #8):** add the SECURITY/PROVENANCE arc (Findings 6–11) — a new results section
  (poisoning ASR, laundering naive-vs-taint, min-cap utility cost, extraction tax) + §2 related-work
  security paragraph (AgentPoison/MINJA as CATEGORY, OWASP ASI06, 2601.05504 as closest prior defense,
  narrowed 4-contribution novelty) + §7 backbone-mismatch caveat (ONE sentence, not a rerun) + §8.
  Watch the abstract/§2 for overclaim creep (the narrowed novelty must survive drafting).
Target venue: agent-memory workshop (NeurIPS/ICLR) or honest arXiv v2.

### After the paper (NOT before): startup is a scoped, security-shaped "conditional yes" (Finding 9/10)
gated on a provenance-integration + hybrid-raw-text-fallback prototype. Not a commitment now.

---

## 6. Working style notes for future sessions
- Be honest and evidence-first; the user explicitly wants flaws surfaced, not hype. Two prior
  LLMs oversold this project; do not repeat that.
- Never paste secrets in chat. The user pasted an OpenRouter key earlier that should be treated
  as burned; the live key is in `sarvam/.env`.
- Prefer offline re-scoring (free) over re-runs (hours). Every eval dumps JSONL for this reason.
- Unverified competitor claims from earlier chats (Microsoft "Memora", "MinnsDB", "Cognee",
  "Bayesian-Agent", specific hackathon dates) are NOT verified — treat with skepticism; only
  BeliefMem (arXiv:2605.05583) is confirmed (it's cited in the paper).
