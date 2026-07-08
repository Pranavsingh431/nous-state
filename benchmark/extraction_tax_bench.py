"""
Extraction-tax measurement — the ONE load-bearing number for the product decision.

Finding 10 showed (n=3) that dropping the pinning costs utility because the real extractor misfiles
or drops facts. This scales that to ~30 real repo facts to get a defensible number: how much clean
utility does nous lose to extraction, with the belief mechanism otherwise perfect?

Design: each fact = a true statement + a question + a short answer, taken from real sarvam/nous
locations (config.py defaults, signature defaults, README). TRUE-ONLY (no poison, no low-trust
channel) so this isolates EXTRACTION quality, nothing else. Two arms, identical render-all-beliefs +
identical LLM answerer:
  pinned    : (entity,attr,value) forced correct, reliability 0.9  -> the ceiling (perfect extraction)
  unpinned  : the REAL extractor decides entity/attr/value from the text (repo tier 0.9, uncapped)
Tax = pinned_accuracy - unpinned_accuracy.

HONEST CAVEAT: render-all-beliefs is GENEROUS to unpinned — a misfiled belief is still in the
context, so the LLM can sometimes recover it. A real deployment would RETRIEVE, and a fact filed
under 'data_persistence' when the query says 'database' would be missed entirely. So this number is a
FLOOR on the tax; real retrieval makes it worse. Domain caveat: repo/technical facts, where
entity/attribute naming is genuinely ambiguous.

Run:  python benchmark/extraction_tax_bench.py     (needs OPENROUTER_API_KEY in ../.env)
"""
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from provenance_agent_bench import llm_answer, load_key, MODEL   # noqa: E402
from provenance_unpinned_bench import render_all_beliefs, extract_claims, _Dummy  # noqa: E402
from nous.engine import Nous          # noqa: E402
from nous.config import config        # noqa: E402
config.ABLATE_UPDATE = ""

# ~30 real repo facts (config.py defaults, signature defaults, README). (entity, attr, value, text, question)
FACTS = [
    ("nous", "graph_max_hops", "2", "The default NOUS_GRAPH_MAX_HOPS is 2.", "What is the default value of GRAPH_MAX_HOPS?"),
    ("nous", "graph_min_confidence", "0.25", "The graph minimum confidence threshold defaults to 0.25.", "What is the default GRAPH_MIN_CONFIDENCE?"),
    ("nous", "graph_max_entities", "15", "GRAPH_MAX_ENTITIES defaults to 15.", "What is the default GRAPH_MAX_ENTITIES?"),
    ("nous", "rerank_top_k", "10", "The reranker keeps the top 10 results by default (RERANK_TOP_K).", "What is the default RERANK_TOP_K?"),
    ("nous", "dense_weight", "0.35", "The dense retrieval weight (DENSE_WEIGHT) is 0.35 by default.", "What is the default DENSE_WEIGHT?"),
    ("nous", "bm25_weight", "0.65", "The BM25 weight is 0.65 by default.", "What is the default BM25 weight?"),
    ("nous", "evidence_limit", "10", "The default evidence limit is 10 items.", "What is the default evidence limit?"),
    ("nous", "evidence_limit_temporal", "15", "For temporal queries the evidence limit is 15.", "What is the temporal evidence limit?"),
    ("nous", "coupling_min_overlap", "3", "Identity coupling requires a minimum overlap of 3 attributes.", "What is the minimum overlap for identity coupling?"),
    ("nous", "coupling_merge_threshold", "0.97", "The coupling merge threshold is 0.97.", "What is the coupling merge threshold?"),
    ("nous", "decay_half_life_days", "30", "The entropy decay half-life is 30 days by default.", "What is the decay half-life in days?"),
    ("nous", "embedding_batch_size", "32", "Embeddings are computed in batches of 32.", "What is the embedding batch size?"),
    ("nous", "embedding_model", "qwen3-embedding-8b", "The embedding model is qwen3-embedding-8b.", "What embedding model does nous use?"),
    ("nous", "novelty_prior", "0.05", "The novelty prior for new values is 0.05.", "What is the novelty prior?"),
    ("nous", "default_model", "gemini-2.5-flash", "The default extraction model is google/gemini-2.5-flash.", "What is the default extraction model?"),
    ("nous", "db_path", "memory.db", "The default db_path is memory.db.", "What is the default db_path filename?"),
    ("nous", "extractor_max_tokens", "4096", "The extractor requests up to 4096 max tokens.", "What is the extractor's max_tokens?"),
    ("nous", "cache_maxsize", "1000", "The internal cache has a maxsize of 1000 entries.", "What is the cache maxsize?"),
    ("nous", "belief_confidence_threshold", "0.80", "answer_from_beliefs uses a confidence threshold of 0.80.", "What confidence threshold does answer_from_beliefs use?"),
    ("nous", "rerank_top_k_env", "10", "NOUS_RERANK_TOP_K controls reranking and defaults to 10.", "How many results does the reranker keep?"),
    ("nous", "database", "SQLite", "The nous library persists its data to SQLite.", "What database does nous persist to?"),
    ("nous", "language", "Python", "Nous is implemented in pure Python.", "What language is nous written in?"),
    ("nous", "license", "MIT", "The project is released under the MIT license.", "What license does nous use?"),
    ("nous", "api_provider", "OpenRouter", "Nous calls LLMs through the OpenRouter API.", "Which API provider does nous use?"),
    ("nous", "runtime_dependencies", "zero", "Nous has zero runtime dependencies; it uses only the Python standard library.", "How many runtime dependencies does nous have?"),
    ("nous", "surprise_unit", "bits", "Surprise is measured in bits, as negative log base 2 of probability.", "In what unit is surprise measured?"),
    ("nous", "core_structure", "dimension", "The core data structure is the dimension, a categorical distribution per entity-attribute pair.", "What is the core data structure in nous called?"),
    ("nous", "stored_artifact", "delta", "The primary stored artifact is the delta, recording the shift from prior to posterior belief.", "What is the primary stored artifact in nous?"),
    ("nous", "forgetting", "entropy decay", "Forgetting in nous is modeled as entropy decay toward the uniform distribution.", "How does nous model forgetting?"),
    ("nous", "update_rule", "Bayesian", "Beliefs are updated with a closed-form Bayesian posterior.", "What kind of update rule does nous use?"),
]


def norm(s):
    return "".join(c for c in str(s).lower() if c.isalnum())

def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def hit(ans, val):
    """Match the short answer value. Handles numeric equivalence (0.8==0.80, 30==30.0)."""
    toks = re.findall(r"[a-z0-9.]+", ans.lower())
    vnum = _num(val)
    if vnum is not None:                                    # numeric: compare by value, not string
        return any(_num(t) is not None and _num(t) == vnum for t in toks)
    v = val.lower()
    if v in set(toks):
        return True
    if v in ans.lower() and (len(norm(val)) >= 4 or "." in val):
        return True
    return norm(val) in norm(ans) and len(norm(val)) >= 4


def make_engine():
    return Nous(db_path=tempfile.NamedTemporaryFile(suffix=".db", delete=False).name, extractor=_Dummy())


def run():
    if not load_key():
        print("ERROR: OPENROUTER_API_KEY not set in ../.env"); sys.exit(1)
    print(f"EXTRACTION TAX — {len(FACTS)} real repo facts, true-only, pinned vs unpinned. Model {MODEL}.")
    print("(render-all answerer -> the number is a FLOOR; real retrieval would raise the tax.)\n")
    print(f"{'#':>2} {'fact':<26}{'answer':>18}{'pin':>7}{'unpin':>8}")
    print("-" * 63)
    pin_ok = unpin_ok = 0
    fails = []
    for i, (e, a, v, text, q) in enumerate(FACTS, 1):
        # pinned: perfect extraction, reliability 0.9 (ceiling)
        ep = make_engine(); ep.observe(text, claims=[(e, a, v, 0.9)])
        pa = llm_answer(render_all_beliefs(ep), q)
        ph = hit(pa, v)
        # unpinned: real extractor decides e/a/v; repo tier 0.9 (uncapped high-trust)
        eu = make_engine()
        capped = [(ce, ca, cv, min(0.9, cc)) for (ce, ca, cv, cc) in extract_claims(text)]
        if capped:
            eu.observe(text, claims=capped)
        ua = llm_answer(render_all_beliefs(eu), q)
        uh = hit(ua, v)
        pin_ok += ph; unpin_ok += uh
        if ph and not uh:
            fails.append((a, v, ua[:24], [f"{ce}.{ca}={cv}" for (ce, ca, cv, _) in capped]))
        print(f"{i:>2} {a:<26}{v:>18}{'OK' if ph else 'X':>7}{'OK' if uh else 'X':>8}")
    n = len(FACTS)
    pin = 100 * pin_ok / n; unp = 100 * unpin_ok / n
    print("-" * 63)
    print(f"{'ACCURACY':<29}{'':>18}{pin:>6.0f}%{unp:>7.0f}%")
    print(f"\nEXTRACTION TAX = {pin - unp:.0f} points  (pinned {pin:.0f}% -> unpinned {unp:.0f}%, n={n})")
    if fails:
        print(f"\nWhere unpinned lost ({len(fails)} facts) — extractor misfiled/dropped the value:")
        for a, v, ans, claims in fails:
            print(f"  {a} (want {v}): answered {ans!r}; extracted {claims or '[nothing]'}")
    print("\nCaveat: floor (render-all is generous; retrieval would miss misfiled beliefs); "
          "repo/technical domain; single run (LLM extraction is mildly non-deterministic).")


if __name__ == "__main__":
    run()
