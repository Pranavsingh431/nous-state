"""
Contradiction / staleness micro-benchmark for the Nous belief mechanism.

Purpose: the LoCoMo ablation showed the Bayesian update is inert vs naive last-write-wins
on standard conversational QA. That is the wrong test — Bayesian belief-tracking should only
help where evidence is CONTRADICTORY, NOISY, and of VARYING RELIABILITY. This benchmark tests
exactly that, at the belief layer (no LLM, no retrieval), so the mechanism is isolated.

Task: feed a sequence of (value, reliability) observations for a single entity.attribute,
then ask for the CURRENT true value. Score = accuracy of each strategy's current best belief.

Strategies compared:
  nous_bayes   Nous BayesianUpdater (closed-form posterior, reliability-weighted)   <- the claim
  flat_lww     last-write-wins (believe the most recent observation)  == ABLATE_UPDATE=flat
  freq         most frequently observed value (ties -> most recent)

Scenario types (each isolates a regime):
  stable_noise    true value V, established; sporadic false low-signal contradictions.
                  Bayesian SHOULD win (flat flips to the noise when noise is last).
  clean_change    V1 established, then genuinely changes to V2 with several V2 obs.
                  Both should get it (sanity).
  recent_change   V1 established, then ONE reliable V2 at the very end.
                  flat SHOULD win (Bayesian lags adopting a fresh change). Tests the weakness.
  noisy_change    V1 -> V2 change with spurious noise mixed around it. Mixed / realistic.
  reliability     correct V asserted by a FEW high-reliability obs; wrong W asserted by MANY
                  low-reliability obs. Only meaningful in the varying-reliability regime:
                  tests whether reliability weighting beats raw frequency/recency.

Regimes:
  constant    every observation has the same reliability (what the real Nous pipeline feeds,
              since observe() uses a fixed default). Bayesian can only exploit repetition.
  varying     reliability reflects true source trust (requires reliability extraction Nous
              does not currently do — so this is the mechanism's optimistic ceiling).

Run:  python benchmark/contradiction_bench.py
Fully offline, deterministic (seeded), no API.
"""

import os
import sys
import random
from collections import Counter, defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure production (non-ablated) Bayesian behaviour regardless of ambient env.
os.environ.pop("NOUS_ABLATE_UPDATE", None)
from nous.dimension import Dimension          # noqa: E402
from nous.updater import BayesianUpdater      # noqa: E402
from nous.config import config                # noqa: E402
config.ABLATE_UPDATE = ""

VALUES = [f"v{i}" for i in range(6)]
N_PER_CELL = 400
SEED = 20260707

updater = BayesianUpdater()


# ---- strategies: each takes a list of (value, reliability), returns predicted current value ----

def predict_nous_bayes(obs):
    dim = Dimension("e.a")
    for val, rel in obs:
        updater.update(dim, val, evidence="", source="s", reliability=rel)
    # ignore the initial 'unknown' seed if a real value dominates
    dist = {k: v for k, v in dim.distribution.items() if k != "unknown"} or dim.distribution
    return max(dist, key=dist.get)


def predict_flat_lww(obs):
    return obs[-1][0]


def predict_freq(obs):
    counts = Counter(v for v, _ in obs)
    best = max(counts.values())
    # tie-break: most recent among the tied
    tied = {v for v, c in counts.items() if c == best}
    for v, _ in reversed(obs):
        if v in tied:
            return v


STRATEGIES = {"nous_bayes": predict_nous_bayes, "flat_lww": predict_flat_lww, "freq": predict_freq}


# ---- scenario generators: return (observations, ground_truth_current_value) ----

def _pick_two(rng):
    a, b = rng.sample(VALUES, 2)
    return a, b


def gen_stable_noise(rng, varying):
    V, W = _pick_two(rng)
    n_true = rng.randint(4, 8)
    n_noise = rng.randint(1, 3)
    hi, lo = (0.9, 0.4) if varying else (0.8, 0.8)
    obs = [(V, hi)] * n_true + [(W, lo)] * n_noise
    rng.shuffle(obs)
    # noise is frequently the most recent event in the real world -> make it so ~half the time
    if rng.random() < 0.5:
        obs = [o for o in obs if o[0] != W] + [(W, lo)] * n_noise
    return obs, V


def gen_clean_change(rng, varying):
    V1, V2 = _pick_two(rng)
    r = 0.9 if varying else 0.8
    obs = [(V1, r)] * rng.randint(3, 6) + [(V2, r)] * rng.randint(3, 6)
    return obs, V2


def gen_recent_change(rng, varying):
    V1, V2 = _pick_two(rng)
    r = 0.9 if varying else 0.8
    obs = [(V1, r)] * rng.randint(4, 8) + [(V2, r)]   # single fresh, reliable change at the end
    return obs, V2


def gen_noisy_change(rng, varying):
    V1, V2 = _pick_two(rng)
    W = rng.choice([v for v in VALUES if v not in (V1, V2)])
    hi, lo = (0.9, 0.4) if varying else (0.8, 0.8)
    obs = [(V1, hi)] * rng.randint(3, 5) + [(V2, hi)] * rng.randint(2, 4) + [(W, lo)] * rng.randint(1, 2)
    # keep a couple of V2 after the noise so the true current value is V2
    obs += [(V2, hi)] * rng.randint(1, 2)
    return obs, V2


def gen_reliability(rng, varying):
    V, W = _pick_two(rng)
    if varying:
        hi, lo = 0.95, 0.3
    else:
        hi, lo = 0.8, 0.8   # in constant regime there is no reliability signal to exploit
    obs = [(V, hi)] * rng.randint(2, 3) + [(W, lo)] * rng.randint(4, 7)
    rng.shuffle(obs)
    return obs, V


SCENARIOS = {
    "stable_noise": gen_stable_noise,
    "clean_change": gen_clean_change,
    "recent_change": gen_recent_change,
    "noisy_change": gen_noisy_change,
    "reliability": gen_reliability,
}


def run():
    rng = random.Random(SEED)
    # acc[regime][scenario][strategy] = [correct, total]
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))

    for regime in ("constant", "varying"):
        varying = regime == "varying"
        for sname, gen in SCENARIOS.items():
            for _ in range(N_PER_CELL):
                obs, gt = gen(rng, varying)
                for stratname, fn in STRATEGIES.items():
                    pred = fn(obs)
                    cell = acc[regime][sname][stratname]
                    cell[0] += int(pred == gt)
                    cell[1] += 1

    for regime in ("constant", "varying"):
        print(f"\n================  REGIME: {regime} reliability  ================")
        header = f"{'scenario':<15}" + "".join(f"{s:>13}" for s in STRATEGIES)
        print(header)
        print("-" * len(header))
        macro = defaultdict(lambda: [0, 0])
        for sname in SCENARIOS:
            row = f"{sname:<15}"
            for stratname in STRATEGIES:
                c, t = acc[regime][sname][stratname]
                row += f"{100 * c / t:>12.1f}%"
                macro[stratname][0] += c
                macro[stratname][1] += t
            print(row)
        print("-" * len(header))
        row = f"{'MACRO':<15}"
        for stratname in STRATEGIES:
            c, t = macro[stratname]
            row += f"{100 * c / t:>12.1f}%"
        print(row)

    print("\nRead: nous_bayes should beat flat_lww on stable_noise / reliability (noise"
          " resistance), and lose on recent_change (adoption lag). Net advantage = whether"
          " the wins outweigh the losses, and only the 'varying' regime gives Bayesian the"
          " reliability signal its theory needs.")


if __name__ == "__main__":
    run()
