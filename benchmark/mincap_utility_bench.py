"""
Benign-UTILITY cost of the provenance min-cap.

Finding 7's defense is trust = min(provenance_tier, content_confidence): content may only ever
LOWER trust within a provenance ceiling, never raise it. That defeats confident poison. But it has
a benign cost: a RELIABLE-but-infrequent source that happens to arrive on a LOW provenance channel
(a knowledgeable "unofficial" source, an expert user vs the official system-of-record) gets its
correct, confident content capped down to the channel's low tier -- and can then be out-voted by
frequent low-confidence noise. This benchmark measures how much of the Finding-5 advantage the cap
destroys, as a function of the true source's provenance tier.

Setup mirrors Finding 5 at the belief layer, using the extractor's MEASURED content reliabilities
(confident -> 0.9, hedged -> 0.25). Two regimes where reliability-weighting was the win:
  reliability_conflict : true V by FEW confident obs; false W by MANY hedged obs. GT=V.
  stable_noise         : true V by MANY confident obs; false W by FEW hedged obs (sometimes last). GT=V.

Strategies:
  content_only        r = content_confidence            (Finding-5 baseline, no provenance cap)
  hybrid_min(t_true)  r = min(t_true, content) for the TRUE source; noise uncapped. Sweep t_true.

Accuracy = fraction of scenarios whose current-belief argmax == GT (the true value). The drop from
content_only to hybrid_min as t_true falls IS the security/utility tradeoff a buyer hits day one.

Run:  python benchmark/mincap_utility_bench.py     (offline, seeded, no API)
"""
import os
import sys
import random
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.pop("NOUS_ABLATE_UPDATE", None)

from nous.dimension import Dimension        # noqa: E402
from nous.updater import BayesianUpdater    # noqa: E402
from nous.config import config              # noqa: E402
config.ABLATE_UPDATE = ""
updater = BayesianUpdater()

VALUES = [f"v{i}" for i in range(4)]
CONF, HEDGE = 0.9, 0.25          # measured content reliabilities (Finding 5 / 7)
NOISE_TIER = 0.9                 # noise arrives on an ordinary channel (uncapped) -> isolates the
                                # cost of capping the TRUE source specifically
N_PER = 400
SEED = 20260709


def argmax_belief(obs):
    dim = Dimension("e.a")
    for val, rel in obs:
        updater.update(dim, val, evidence="", source="s", reliability=rel)
    dist = {k: v for k, v in dim.distribution.items() if k != "unknown"} or dim.distribution
    return max(dist, key=dist.get)


def gen_reliability_conflict(rng):
    V, W = rng.sample(VALUES, 2)
    true_obs = [(V, CONF)] * rng.randint(2, 3)       # (value, content_conf)
    noise_obs = [(W, HEDGE)] * rng.randint(4, 6)
    seq = true_obs + noise_obs
    rng.shuffle(seq)
    return seq, V, W


def gen_stable_noise(rng):
    V, W = rng.sample(VALUES, 2)
    true_obs = [(V, CONF)] * rng.randint(4, 6)
    noise_obs = [(W, HEDGE)] * rng.randint(1, 2)
    if rng.random() < 0.5:
        seq = true_obs + noise_obs                    # noise last
    else:
        seq = true_obs + noise_obs; rng.shuffle(seq)
    return seq, V, W


REGIMES = {"reliability_conflict": gen_reliability_conflict, "stable_noise": gen_stable_noise}


def effective(seq, V, W, strategy, t_true):
    """Map (value, content_conf) obs -> (value, effective_reliability) under a strategy."""
    out = []
    for val, conf in seq:
        if strategy == "content_only":
            r = conf
        else:  # hybrid_min: cap the TRUE source by its (low) provenance tier; leave noise uncapped
            if val == V:
                r = min(t_true, conf)
            else:
                r = min(NOISE_TIER, conf)
        out.append((val, r))
    return out


def accuracy(regime_gen, strategy, t_true, rng):
    hits = 0
    for _ in range(N_PER):
        seq, V, W = regime_gen(rng)
        if argmax_belief(effective(seq, V, W, strategy, t_true)) == V:
            hits += 1
    return 100.0 * hits / N_PER


def main():
    tiers = [0.9, 0.7, 0.5, 0.3, 0.2]
    print("=== BENIGN-UTILITY COST OF THE min-CAP ===")
    print("Accuracy (current belief == true value). content_only = Finding-5 baseline (no cap).")
    print("hybrid_min routes the RELIABLE true source through provenance tier t_true and caps it.\n")
    for regime, gen in REGIMES.items():
        rng = random.Random(SEED)
        base = accuracy(gen, "content_only", 1.0, rng)
        print(f"-- {regime} --")
        print(f"   content_only (no provenance cap): {base:.1f}%")
        hdr = "   hybrid_min t_true:" + "".join(f"{t:>8}" for t in tiers)
        print(hdr)
        row = "   accuracy:         " + "".join(f"{accuracy(gen, 'hybrid_min', t, rng):>7.1f}%" for t in tiers)
        print(row)
        print()
    print("Reading: where accuracy stays near content_only, the cap is free. Where it collapses as")
    print("t_true falls, that is the utility you pay for poisoning safety: correct, confident info")
    print("on a low-trust channel gets discounted and out-voted by frequent low-confidence noise.")
    print("Design implication: provenance tiers must be assigned so genuinely-reliable sources are")
    print("NOT stuck in low tiers -- otherwise the security cap silently degrades benign accuracy.")


if __name__ == "__main__":
    main()
