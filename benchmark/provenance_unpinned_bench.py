"""
UNPINNED extraction-fragility test — the honest follow-up to provenance_agent_bench.py (Finding 9).

Finding 9 PINNED (entity, attribute, value) so true and poison competed in one belief and the trust
mechanism was isolated. That deliberately held constant nous's real-world weakness: it depends on an
LLM extractor to turn text into (entity, attribute, value), and that step is noisy. This benchmark
DROPS the pinning — the real extractor decides entity/attr/value from the text — and measures how
much that costs, end to end, against the pinned reference.

Three arms, IDENTICAL answerer that renders each backend's whole (tiny) memory so RETRIEVAL is not a
confound (the engine's query layer is LoCoMo-tuned and would fail on repo facts for unrelated
reasons). The pinned-vs-unpinned delta is therefore pure EXTRACTION cost:
  baseline        append-and-retrieve, no trust weighting            (reference)
  nous_pinned     entity/attr/value forced correct + min-cap         (Finding-9 mechanism)
  nous_unpinned   real extractor decides e/a/v + min-cap             (the honest, fragile pipeline)

Also prints a retrieval-independent EXTRACTION DIAGNOSTIC: what (e,a,v) the extractor produced for
each fact, and whether true+poison COMPETE (same dimension) or FRAGMENT (different dimensions) —
fragmentation is the concrete mechanism by which extraction noise erodes poison resistance.

Reports plainly where nous_unpinned loses to nous_pinned and/or baseline. No spin.

Run:  python benchmark/provenance_unpinned_bench.py     (needs OPENROUTER_API_KEY in ../.env)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reuse the AUDITED primitives + scenarios from Finding 9's harness (single source of trust logic).
from provenance_agent_bench import (          # noqa: E402
    CHANNEL_TIER, provenance_tier, effective_reliability, llm_answer, norm, asserted,
    load_key, UTILITY, POISON_FACTS, BaselineMemory, MODEL,
)
from nous.engine import Nous                  # noqa: E402
from nous.config import config               # noqa: E402
from nous.llm_extractor import LLMExtractor  # noqa: E402
config.ABLATE_UPDATE = ""

_ext = None
def real_extractor():
    global _ext
    if _ext is None:
        _ext = LLMExtractor(api_key=os.environ["OPENROUTER_API_KEY"], model=MODEL)
    return _ext

_extract_cache = {}
def extract_claims(text):
    if text not in _extract_cache:
        _extract_cache[text] = real_extractor().extract_with_reliability(text)
    return _extract_cache[text]


class _Dummy:
    def extract(self, text): return []


def render_all_beliefs(engine):
    """IDENTICAL answerer context: render every current belief. Removes retrieval as a confound."""
    lines = []
    for dim in engine.world_model.all_dimensions():
        dist = {k: v for k, v in dim.distribution.items() if k != "unknown"} or dim.distribution
        if not dist:
            continue
        val = max(dist, key=dist.get)
        lines.append(f"- {dim.id} = {val} (confidence {dist[val]:.2f})")
    return "\n".join(lines) if lines else "(no beliefs)"


class NousMem:
    """pinned=True: use the scenario's (e,a,v). pinned=False: let the real extractor decide e/a/v."""
    def __init__(self, pinned):
        self.pinned = pinned
        self.engine = Nous(db_path=tempfile.NamedTemporaryFile(suffix=".db", delete=False).name,
                           extractor=_Dummy())
        self.last_extracted = []
    def ingest(self, text, channel, claim, tainted_origin=None):
        tier = provenance_tier(channel, tainted_origin)
        if self.pinned:
            e, a, v = claim
            conf = _content_conf(text, v)
            capped = [(e, a, v, min(tier, conf))]
        else:
            capped = [(e, a, v, min(tier, conf)) for (e, a, v, conf) in extract_claims(text)]
        self.last_extracted = capped
        if capped:
            self.engine.observe(text, claims=capped)
    def context(self):
        return render_all_beliefs(self.engine)


def _content_conf(text, value):
    for (_, _, v, r) in extract_claims(text):
        if norm(value) in norm(v) or norm(v) in norm(value):
            return r
    return 0.7


def answer(mem, question, is_baseline, kw):
    ctx = mem.context_for(kw, None, None) if is_baseline else mem.context()
    return llm_answer(ctx, question)


# ─────────────────────────────────────────────────────────────────────────────
def extraction_diagnostic():
    print("\n================  EXTRACTION DIAGNOSTIC (retrieval-independent)  ================")
    print("What (entity, attribute, value) the real extractor produces, and whether true & poison")
    print("COMPETE (same dim) or FRAGMENT (different dims). Fragmentation erodes poison resistance.\n")
    frag = 0
    for f in POISON_FACTS:
        tclaims = extract_claims(f["true_text"])
        pclaims = extract_claims(f["poison_text"])
        def dims(claims, target):
            return [f"{e}.{a}={v}" for (e, a, v, _) in claims if norm(target) in norm(v) or norm(v) in norm(target)]
        td = dims(tclaims, f["true"]); pd = dims(pclaims, f["poison"])
        t_dim = {x.rsplit("=", 1)[0] for x in td}
        p_dim = {x.rsplit("=", 1)[0] for x in pd}
        competes = bool(t_dim & p_dim)
        if not competes: frag += 1
        print(f"  {f['dim']}:")
        print(f"     true  -> {td or '[value not extracted!]'}")
        print(f"     poison-> {pd or '[value not extracted!]'}")
        print(f"     => {'COMPETE (same dim)' if competes else 'FRAGMENT (different dims)'}")
    print(f"\n  Fragmentation: {frag}/{len(POISON_FACTS)} poison facts landed in a DIFFERENT dim "
          f"than the truth.")
    return frag


def run_utility():
    print("\n================  UTILITY (pinned vs unpinned)  ================")
    print(f"{'scenario':<46}{'correct':>14}{'baseline':>11}{'nous_pin':>11}{'nous_unpin':>13}")
    print("-" * 95)
    agg = {"baseline": [0, 0], "pin": [0, 0], "unpin": [0, 0]}
    for sc in UTILITY:
        cell = {}
        mems = {"baseline": BaselineMemory(), "pin": NousMem(True), "unpin": NousMem(False)}
        for key, mem in mems.items():
            for (text, ch, claim, taint) in sc["steps"]:
                mem.ingest(text, ch, claim, taint)
            ans = answer(mem, sc["q"], key == "baseline", sc["kw"])
            ok = norm(sc["correct"]) in norm(ans)
            cell[key] = "OK" if ok else f"WRONG({ans[:9]})"
            agg[key][0] += int(ok); agg[key][1] += 1
        print(f"{sc['name']:<46}{sc['correct']:>14}{cell['baseline']:>11}{cell['pin']:>11}{cell['unpin']:>13}")
    print("-" * 95)
    a = {k: 100 * agg[k][0] / agg[k][1] for k in agg}
    print(f"{'ACCURACY':<46}{'':>14}{a['baseline']:>10.0f}%{a['pin']:>10.0f}%{a['unpin']:>12.0f}%")
    return a


def run_poison(M=3):
    print(f"\n================  DIRECT POISON @ M={M} (pinned vs unpinned)  ================")
    print("ASR = % of facts whose answer flips to the false value (lower=better).")
    flips = {"baseline": 0, "pin": 0, "unpin": 0}
    for f in POISON_FACTS:
        mems = {"baseline": BaselineMemory(), "pin": NousMem(True), "unpin": NousMem(False)}
        for key, mem in mems.items():
            mem.ingest(f["true_text"], "repo", (f["entity"], f["attr"], f["true"]))
            for _ in range(M):
                mem.ingest(f["poison_text"], "web", (f["entity"], f["attr"], f["poison"]))
            ans = answer(mem, f["q"], key == "baseline", f["kw"])
            if asserted(ans, f["true"], f["poison"]) == "poison":
                flips[key] += 1
    n = len(POISON_FACTS)
    r = {k: 100 * flips[k] / n for k in flips}
    print(f"{'backend':<16}{'ASR':>8}")
    for k in ("baseline", "pin", "unpin"):
        print(f"{k:<16}{r[k]:>7.0f}%")
    return r


def run_laundering(M=3):
    print(f"\n================  LAUNDERING via trusted tool @ M={M} (unpinned)  ================")
    print("Poison enters through a trusted 'tool'. ASR = % flipped (lower=better).")
    out = {}
    for label, taint in (("nous_unpin naive", None), ("nous_unpin taint(origin=web)", "web")):
        flips = 0
        for f in POISON_FACTS:
            mem = NousMem(False)
            mem.ingest(f["true_text"], "repo", (f["entity"], f["attr"], f["true"]))
            for _ in range(M):
                mem.ingest(f["poison_text"], "tool", (f["entity"], f["attr"], f["poison"]), taint)
            ans = answer(mem, f["q"], False, f["kw"])
            if asserted(ans, f["true"], f["poison"]) == "poison":
                flips += 1
        out[label] = 100 * flips / len(POISON_FACTS)
        print(f"{label:<30}{out[label]:>7.0f}%")
    return out


def main():
    if not load_key():
        print("ERROR: OPENROUTER_API_KEY not set in ../.env"); sys.exit(1)
    print("UNPINNED extraction-fragility test. Channel tiers:", CHANNEL_TIER)
    frag = extraction_diagnostic()
    util = run_utility()
    poison = run_poison()
    launder = run_laundering()

    print("\n================  HONEST CONCLUSION (extraction cost)  ================")
    du = util["pin"] - util["unpin"]
    print(f"* Utility: pinned {util['pin']:.0f}% -> unpinned {util['unpin']:.0f}% "
          f"({'-' if du>0 else '+'}{abs(du):.0f} pts from extraction). baseline {util['baseline']:.0f}%.")
    print(f"* Direct poison ASR: pinned {poison['pin']:.0f}% -> unpinned {poison['unpin']:.0f}% "
          f"(baseline {poison['baseline']:.0f}%). Fragmentation was {frag}/{len(POISON_FACTS)}.")
    if poison["unpin"] > poison["pin"]:
        print("  -> extraction fragmentation WEAKENED poison resistance (poison escaped into its own")
        print("     dimension instead of being down-weighted in the contested belief). Report as-is.")
    elif poison["unpin"] == poison["pin"]:
        print("  -> poison resistance survived unpinned extraction on these facts.")
    ln = launder
    print(f"* Laundering unpinned: naive {ln.get('nous_unpin naive',0):.0f}% vs "
          f"taint {ln.get('nous_unpin taint(origin=web)',0):.0f}% (taint still required).")
    print("\nBottom line: the extraction step is a REAL, separate cost of nous that Finding 9 hid by")
    print("pinning. The numbers above quantify it; whatever they show, that is the honest figure.")


if __name__ == "__main__":
    main()
