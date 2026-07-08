"""
End-to-end reliability benchmark for the Nous belief mechanism.

WHY THIS EXISTS
---------------
Finding 2: on LoCoMo the Bayesian update is INERT — naive last-write-wins ties it.
Finding 4 (belief layer, no LLM): the Bayesian update only beats last-write-wins when
observations carry VARYING reliability; with the constant reliability the real pipeline feeds,
the posterior degenerates to a soft recency-follower. The named fix is RELIABILITY EXTRACTION:
estimate a per-observation reliability from the speaker's epistemic markers and let the Bayesian
update weight evidence by it. This benchmark tests whether that fix works END TO END, through the
real extractor + engine, on natural-language evidence — the regime LoCoMo never exercises.

TASK
----
For each scenario, an entity's attribute (e.g. Marco.employer) evolves over time via a sequence
of natural-language statements. Statements vary in epistemic confidence (confident / hedged /
corrective) expressed ONLY in phrasing — never labeled in the text the extractor sees. Some
statements are false (noise). After all statements, we ask for the CURRENT true value.

STRATEGIES (all read the same designed (entity, attribute, value); they differ only in how the
belief is formed):
  nous_reliability  Bayesian posterior, per-claim reliability from extract_with_reliability()   <- the claim
  nous_flat         Bayesian update replaced by last-write-wins (NOUS_ABLATE_UPDATE=flat)
  freq              most-frequent value (recency tie-break) — the simplest non-Bayesian baseline
  append_retrieve   LLM reads all statements in order and answers (the vector/append-memory baseline; live only)

RELIABILITY ISOLATION
---------------------
To isolate the scientific question (does reliability-weighting help?) from general extraction
noise, entity/attribute/value are PINNED to the scenario design; only the per-claim reliability
comes from the LLM (matched to the designed value). This mirrors how contradiction_bench isolated
the belief layer. Stage-A output reports whether the LLM's reliability actually tracks the intended
confidence — if it doesn't, the whole approach is moot and the live numbers will show it.

HONESTY GUARDS
--------------
- An `adversarial` scenario type inverts the confidence/correctness correlation (the TRUE value is
  hedged, a FALSE value is asserted confidently). Reliability-weighting SHOULD lose there. It is
  reported as its own row so the win is never laundered across a rigged average.
- --mock replaces the LLM with perfect reliability (from the intended level) to validate scenario
  logic, scoring, and strategy replay offline for $0 before any live run.

Run:
  python benchmark/reliability_bench.py --mock            # offline, deterministic, no API
  python benchmark/reliability_bench.py                   # live extractor (needs OPENROUTER_API_KEY)
  python benchmark/reliability_bench.py --with-retrieval  # also run the append_retrieve LLM baseline
"""
import os
import sys
import json
import argparse
import random
import tempfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nous.config import config              # noqa: E402
from nous.engine import Nous                # noqa: E402
from nous.llm_extractor import LLMExtractor # noqa: E402

def _load_env_key():
    """Export OPENROUTER_API_KEY from ../.env into os.environ if not already set, so BOTH the
    extractor and every engine instance see it (the engine reads os.environ, not .env). Without
    this the engine harmlessly logs 'not found' and falls back to BM25 — irrelevant to this
    benchmark (it never uses engine retrieval), but confusing. This removes the noise."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    envp = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(envp):
        for line in open(envp):
            if line.strip().startswith("OPENROUTER_API_KEY"):
                os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")


_load_env_key()

SEED = 20260708
N_PER_TYPE = 12

# ---- value pools + natural fact clauses per attribute ----
POOLS = {
    "employer": (["Google", "Meta", "Amazon", "Netflix", "Stripe", "Airbnb", "Datadog", "Snowflake"],
                 "{name} works at {val}"),
    "city":     (["Boston", "Denver", "Austin", "Seattle", "Chicago", "Portland", "Atlanta", "Nashville"],
                 "{name} lives in {val}"),
    "role":     (["Engineer", "Manager", "Designer", "Analyst", "Researcher", "Director"],
                 "{name}'s role is {val}"),
}
NAMES = ["Marco", "Priya", "Diego", "Lena", "Omar", "Sofia", "Kenji", "Nadia",
         "Tomas", "Aisha", "Viktor", "Mei", "Ravi", "Elena", "Hugo", "Yara"]

# Confidence surface forms. {clause} = the fact clause (starts with the name).
# Intended reliability bands mirror the extractor rubric; used by --mock and Stage-A checking.
TEMPLATES = {
    "high": (0.85, [
        "{clause} now — it's official.",
        "To be clear, {clause}. Confirmed today.",
        "{clause}, definitely.",
        "Actually, {clause} — I just confirmed it.",
        "For sure, {clause}.",
    ]),
    "mid": (0.6, [
        "I'm pretty sure {clause}.",
        "{clause}, I believe.",
        "I think {clause}.",
    ]),
    "low": (0.25, [
        "Maybe {clause}? Not totally sure.",
        "{clause}, but I could be wrong.",
        "I heard {clause}, no idea if it's true.",
        "{clause}? I might be misremembering.",
    ]),
}

# HARD phrasing bank (RELIABILITY_HARD=1): a much larger, more naturalistic set with SUBTLE hedges
# that the extractor rubric never lists ("apparently", "supposedly", "last I checked", "if I
# recall"...). This tests whether the extractor reads epistemic STANCE vs pattern-matching the 5
# canned forms above — the main robustness critique of the templated benchmark.
TEMPLATES_HARD = {
    "high": (0.85, [
        "{clause} now — it's official.",
        "To be clear, {clause}. Confirmed today.",
        "{clause}, definitely.",
        "Actually, {clause} — I just confirmed it.",
        "For sure, {clause}.",
        "No doubt about it, {clause}.",
        "Just to set the record straight: {clause}.",
        "100%, {clause}.",
        "{clause} — signed the papers this morning.",
        "I can confirm firsthand that {clause}.",
        "{clause}, and that's final.",
        "Believe me, {clause}.",
    ]),
    "mid": (0.6, [
        "I'm pretty sure {clause}.",
        "{clause}, I believe.",
        "I think {clause}.",
        "As far as I know, {clause}.",
        "{clause}, if I'm not mistaken.",
        "Fairly confident {clause}.",
    ]),
    "low": (0.25, [
        "Maybe {clause}? Not totally sure.",
        "{clause}, but I could be wrong.",
        "I heard {clause}, no idea if it's true.",
        "{clause}? I might be misremembering.",
        "Apparently {clause}, or so someone said.",
        "Last I checked {clause}, though that was ages ago.",
        "If I recall, {clause}, but it's fuzzy.",
        "Supposedly {clause}.",
        "{clause}, I guess? Hard to say.",
        "Don't quote me, but {clause}.",
        "Rumor has it {clause}.",
        "{clause}, maybe — my memory's hazy on this.",
    ]),
}

if os.environ.get("RELIABILITY_HARD") == "1":
    TEMPLATES = TEMPLATES_HARD


def _utt(rng, name, attr, val, level):
    clause = POOLS[attr][1].format(name=name, val=val)
    band, forms = TEMPLATES[level]
    text = rng.choice(forms).format(clause=clause)
    return {"text": text, "entity": name, "attr": attr, "value": val,
            "level": level, "intended_rel": band}


def _scaffold(rng):
    name = rng.choice(NAMES)
    attr = rng.choice(list(POOLS))
    vals = rng.sample(POOLS[attr][0], 3)
    return name, attr, vals


# ---- scenario generators: return (utterances, ground_truth_value, type) ----

def gen_stable_noise(rng):
    """True V asserted confidently; a false W asserted with LOW confidence, sometimes last.
    Reliability should down-weight the hedged noise; flat-LWW flips when noise is last."""
    name, attr, (V, W, _) = _scaffold(rng)
    utts = [_utt(rng, name, attr, V, "high") for _ in range(rng.randint(4, 6))]
    noise = [_utt(rng, name, attr, W, "low") for _ in range(rng.randint(1, 2))]
    if rng.random() < 0.5:                      # noise arrives last ~half the time
        rng.shuffle(utts)
        utts = utts + noise
    else:
        utts = utts + noise
        rng.shuffle(utts)
    return utts, V, "stable_noise"


def gen_clean_change(rng):
    """V1 confident, then genuinely changes to V2 confident. Sanity: everyone should get V2."""
    name, attr, (V1, V2, _) = _scaffold(rng)
    utts = [_utt(rng, name, attr, V1, "high") for _ in range(rng.randint(2, 4))]
    utts += [_utt(rng, name, attr, V2, "high") for _ in range(rng.randint(2, 4))]
    return utts, V2, "clean_change"


def gen_recent_change(rng):
    """V1 confident several times, then ONE confident V2 at the very end. GT=V2.
    The mechanism's known weakness (adoption lag): accumulated V1 mass may resist one fresh obs."""
    name, attr, (V1, V2, _) = _scaffold(rng)
    utts = [_utt(rng, name, attr, V1, "high") for _ in range(rng.randint(4, 6))]
    utts += [_utt(rng, name, attr, V2, "high")]
    return utts, V2, "recent_change"


def gen_reliability_conflict(rng):
    """Correct V by FEW high-confidence statements; wrong W by MANY low-confidence ones.
    Frequency & (often) recency favor W; only reliability-weighting recovers V. GT=V."""
    name, attr, (V, W, _) = _scaffold(rng)
    utts = [_utt(rng, name, attr, V, "high") for _ in range(rng.randint(2, 3))]
    utts += [_utt(rng, name, attr, W, "low") for _ in range(rng.randint(4, 6))]
    rng.shuffle(utts)
    return utts, V, "reliability_conflict"


def gen_adversarial(rng):
    """HONESTY GUARD: confidence is INVERTED. True V is hedged; false W is confident. GT=V.
    Reliability-weighting SHOULD lose here (it trusts confidence). Measures the downside."""
    name, attr, (V, W, _) = _scaffold(rng)
    utts = [_utt(rng, name, attr, V, "low") for _ in range(rng.randint(2, 3))]
    utts += [_utt(rng, name, attr, W, "high") for _ in range(rng.randint(2, 3))]
    rng.shuffle(utts)
    return utts, V, "adversarial"


GENERATORS = [gen_stable_noise, gen_clean_change, gen_recent_change,
              gen_reliability_conflict, gen_adversarial]


def build_scenarios():
    rng = random.Random(SEED)
    scenarios = []
    for gen in GENERATORS:
        for _ in range(N_PER_TYPE):
            utts, gt, typ = gen(rng)
            scenarios.append({"utterances": utts, "gt": gt, "type": typ})
    return scenarios


# ---- value matching ----
def norm(s):
    return "".join(ch for ch in s.lower().strip() if ch.isalnum())


def match(pred, gt):
    p, g = norm(pred), norm(gt)
    return p == g or (len(g) >= 4 and (g in p or p in g))


# ---- reliability sources ----
def mock_reliability(rng, level):
    band = TEMPLATES[level][0]
    return min(1.0, max(0.0, rng.gauss(band, 0.05)))


def live_reliability(extractor, utt):
    """Call extract_with_reliability; return the LLM reliability for the claim whose value matches
    the designed value (fallback 0.7 if none matches). Entity/attr/value stay pinned to the design."""
    claims = extractor.extract_with_reliability(utt["text"])
    for (_, _, val, rel) in claims:
        if match(val, utt["value"]):
            return rel
    # try attribute-agnostic: any claim whose value matches
    return 0.7


# ---- belief-layer strategies (read designed dimension after replaying claims) ----
def _fresh_engine():
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    return Nous(db_path=db, extractor=_Dummy())


class _Dummy:
    def extract(self, text):
        return []


def belief_argmax(engine, entity, attr):
    dim = engine.world_model.get_dimension(f"{entity}.{attr}")
    dist = {k: v for k, v in dim.distribution.items() if k != "unknown"} or dim.distribution
    return max(dist, key=dist.get)


def predict_nous(scenario, rels, flat):
    """Replay (entity, attr, value, reliability) claims under Bayesian or flat update; read argmax."""
    prev = config.ABLATE_UPDATE
    config.ABLATE_UPDATE = "flat" if flat else ""
    try:
        eng = _fresh_engine()
        e = scenario["utterances"][0]["entity"]
        a = scenario["utterances"][0]["attr"]
        for utt, rel in zip(scenario["utterances"], rels):
            eng.observe(utt["text"], claims=[(utt["entity"], utt["attr"], utt["value"], rel)])
        return belief_argmax(eng, e, a)
    finally:
        config.ABLATE_UPDATE = prev


def predict_freq(scenario):
    counts = Counter(u["value"] for u in scenario["utterances"])
    best = max(counts.values())
    tied = {v for v, c in counts.items() if c == best}
    for u in reversed(scenario["utterances"]):
        if u["value"] in tied:
            return u["value"]


def predict_append_retrieve(extractor, scenario):
    """Vector/append baseline: LLM reads all statements in order, answers the current value."""
    e = scenario["utterances"][0]["entity"]
    a = scenario["utterances"][0]["attr"]
    lines = "\n".join(f"{i+1}. {u['text']}" for i, u in enumerate(scenario["utterances"]))
    sys_p = ("You are a memory system. Given statements in chronological order (some may be "
             "uncertain or wrong), determine the CURRENT true value. Answer with ONLY the value, "
             "no explanation.")
    q = f"Statements:\n{lines}\n\nWhat is {e}'s current {a.replace('_', ' ')}?"
    out = extractor._chat(sys_p, q, max_tokens=30)
    if isinstance(out, list) and out:
        return str(out[0])
    # _chat expects JSON; fall back to a plain call via a tiny wrapper
    return _plain_answer(extractor, sys_p, q)


def _plain_answer(extractor, sys_p, user_q):
    import urllib.request
    payload = json.dumps({
        "model": extractor.model,
        "messages": [{"role": "system", "content": sys_p + " Respond with just the value word."},
                     {"role": "user", "content": user_q}],
        "temperature": 0.0, "max_tokens": 20,
    }).encode()
    req = urllib.request.Request(f"{extractor.base_url}/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {extractor.api_key}"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            body = json.loads(r.read().decode())
        return body["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def run(mock=False, with_retrieval=False):
    scenarios = build_scenarios()
    rng = random.Random(SEED + 1)

    extractor = None
    if not mock or with_retrieval:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            # load .env
            envp = os.path.join(os.path.dirname(__file__), "..", ".env")
            if os.path.exists(envp):
                for line in open(envp):
                    if line.strip().startswith("OPENROUTER_API_KEY"):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not key and not mock:
            print("ERROR: OPENROUTER_API_KEY not set (needed for live run). Use --mock for offline.")
            sys.exit(1)
        if key:
            extractor = LLMExtractor(api_key=key)

    # ---- reliability extraction (also Stage-A data) ----
    print(f"Reliability source: {'MOCK (perfect)' if mock else 'LIVE extractor'}")
    stage_a = defaultdict(list)   # intended level -> [llm reliabilities]
    all_rels = []
    for si, sc in enumerate(scenarios):
        rels = []
        for utt in sc["utterances"]:
            if mock:
                rel = mock_reliability(rng, utt["level"])
            else:
                rel = live_reliability(extractor, utt)
            rels.append(rel)
            stage_a[utt["level"]].append(rel)
        all_rels.append(rels)
        if not mock and (si + 1) % 10 == 0:
            print(f"  extracted {si+1}/{len(scenarios)} scenarios")

    # ---- Stage A report ----
    print("\n================ STAGE A: does reliability track intended confidence? ================")
    print(f"{'intended level':<16}{'n':>5}{'mean LLM rel':>14}{'(target band)':>16}")
    for level in ("high", "mid", "low"):
        xs = stage_a[level]
        if xs:
            print(f"{level:<16}{len(xs):>5}{sum(xs)/len(xs):>14.3f}{TEMPLATES[level][0]:>16.2f}")
    hi = sum(stage_a["high"]) / max(1, len(stage_a["high"]))
    lo = sum(stage_a["low"]) / max(1, len(stage_a["low"]))
    print(f"\nSeparation (mean high - mean low) = {hi - lo:+.3f}  "
          f"(need clearly > 0 for reliability-weighting to have signal)")

    # ---- strategy predictions ----
    strategies = ["nous_reliability", "nous_flat", "freq"]
    if with_retrieval and extractor:
        strategies.append("append_retrieve")
    # acc[type][strategy] = [correct, total]
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for sc, rels in zip(scenarios, all_rels):
        preds = {
            "nous_reliability": predict_nous(sc, rels, flat=False),
            "nous_flat":        predict_nous(sc, rels, flat=True),
            "freq":             predict_freq(sc),
        }
        if "append_retrieve" in strategies:
            preds["append_retrieve"] = predict_append_retrieve(extractor, sc)
        for strat in strategies:
            cell = acc[sc["type"]][strat]
            cell[0] += int(match(str(preds[strat]), sc["gt"]))
            cell[1] += 1

    # ---- results ----
    print("\n================ RESULTS: accuracy predicting the CURRENT value ================")
    types = [g(random.Random(0))[2] for g in GENERATORS]  # ordered type names
    header = f"{'scenario type':<22}" + "".join(f"{s:>18}" for s in strategies)
    print(header)
    print("-" * len(header))
    macro = defaultdict(lambda: [0, 0])
    macro_real = defaultdict(lambda: [0, 0])   # excludes the adversarial (inverted) slice
    for typ in types:
        row = f"{typ:<22}"
        for strat in strategies:
            c, t = acc[typ][strat]
            row += f"{(100*c/t if t else 0):>17.1f}%"
            macro[strat][0] += c; macro[strat][1] += t
            if typ != "adversarial":
                macro_real[strat][0] += c; macro_real[strat][1] += t
        print(row)
    print("-" * len(header))
    for label, m in (("MACRO (all)", macro), ("MACRO (realistic, no adversarial)", macro_real)):
        row = f"{label:<22}"
        for strat in strategies:
            c, t = m[strat]
            row += f"{(100*c/t if t else 0):>17.1f}%"
        print(row)

    print("\nRead: nous_reliability should beat nous_flat/freq on stable_noise & reliability_conflict")
    print("(noise/low-reliability resistance), roughly tie on clean_change, and LOSE on adversarial")
    print("(inverted confidence). The realistic macro is the honest headline; adversarial is the")
    print("stated failure mode. If nous_reliability ~ nous_flat everywhere, the mechanism is inert.")

    # save raw for offline re-inspection
    out = os.path.join(os.path.dirname(__file__),
                       f"reliability_bench_{'mock' if mock else 'live'}.json")
    with open(out, "w") as f:
        json.dump({"acc": {k: dict(v) for k, v in acc.items()},
                   "stage_a": {k: v for k, v in stage_a.items()}}, f, indent=2)
    print(f"\nSaved raw results -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="offline, perfect reliability (no API)")
    ap.add_argument("--with-retrieval", action="store_true", help="also run append_retrieve LLM baseline")
    args = ap.parse_args()
    run(mock=args.mock, with_retrieval=args.with_retrieval)
