"""
Memory-poisoning benchmark for the Nous belief engine (belief layer, offline, deterministic).

THREAT MODEL (MINJA / AgentPoison-style, reduced to the belief layer so the mechanism is
isolated and the result is not confounded by retrieval or an LLM):

  1. A true belief is ESTABLISHED: an entity.attribute is asserted K times with value V by
     TRUSTED sources (reliability r_legit).
  2. An ATTACKER then INJECTS the belief: M observations of a FALSE target value W arrive
     through a COMPROMISED channel of trust r_poison. The payload carries no detectable
     malicious pattern -- it is just an ordinary observation asserting W.
  3. At query time we read the current best belief. The attack SUCCEEDS if argmax == W.

We report Attack Success Rate (ASR, lower is better) for four memory strategies:
  nous_trust    Bayesian posterior, reliability = SOURCE TRUST (legit r_legit, poison r_poison)
  nous_notrust  Bayesian posterior, but poison is trusted equally (r_legit) -- the ablation that
                shows whether SOURCE TRUST (not the Bayesian math alone) is what defends.
  lww           last-write-wins (naive overwrite)
  majority      most-frequent value (append-and-count, ties -> most recent)

The honest questions this answers:
  (a) Does source-trust weighting reduce ASR where naive memory (lww/majority) fails?
  (b) At what INJECTION VOLUME M does a low-trust flood still overwhelm the true belief?
  (c) What happens when the attacker can forge HIGH trust (r_poison -> r_legit)? (must show it
      fails there -- reliability-weighting cannot defend content it wrongly trusts.)

Run:  python benchmark/poison_bench.py         (fully offline, seeded, no API)
"""
import os
import sys
import random
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.pop("NOUS_ABLATE_UPDATE", None)                # ensure real Bayesian update

from nous.dimension import Dimension        # noqa: E402
from nous.updater import BayesianUpdater    # noqa: E402
from nous.config import config              # noqa: E402
config.ABLATE_UPDATE = ""

updater = BayesianUpdater()

VALUES = [f"v{i}" for i in range(4)]         # k=4 possible values per attribute
K_LEGIT = 4                                  # trusted observations establishing the true belief
R_LEGIT = 0.9
N_TRIALS = 400
SEED = 20260709


def _argmax_belief(obs):
    """obs = list of (value, reliability). Return Bayesian argmax (ignoring the 'unknown' seed)."""
    dim = Dimension("e.a")
    for val, rel in obs:
        updater.update(dim, val, evidence="", source="s", reliability=rel)
    dist = {k: v for k, v in dim.distribution.items() if k != "unknown"} or dim.distribution
    return max(dist, key=dist.get)


def predict(strategy, legit, poison, r_poison):
    """legit/poison are lists of values; return predicted current value under `strategy`."""
    if strategy == "nous_trust":
        obs = [(v, R_LEGIT) for v in legit] + [(w, r_poison) for w in poison]
        return _argmax_belief(obs)
    if strategy == "nous_notrust":
        obs = [(v, R_LEGIT) for v in legit] + [(w, R_LEGIT) for w in poison]  # poison wrongly trusted
        return _argmax_belief(obs)
    if strategy == "lww":
        seq = legit + poison
        return seq[-1]
    if strategy == "majority":
        seq = legit + poison
        c = Counter(seq); best = max(c.values())
        tied = {v for v, n in c.items() if n == best}
        for v in reversed(seq):
            if v in tied:
                return v
    raise ValueError(strategy)


STRATEGIES = ["nous_trust", "nous_notrust", "lww", "majority"]


def asr_cell(rng, M, r_poison):
    """Attack success rate for each strategy at (injection volume M, channel trust r_poison)."""
    hits = defaultdict(int)
    for _ in range(N_TRIALS):
        V, W = rng.sample(VALUES, 2)                  # true value V, attacker target W
        legit = [V] * K_LEGIT
        poison = [W] * M
        for s in STRATEGIES:
            if predict(s, legit, poison, r_poison) == W:   # attacker's target adopted -> success
                hits[s] += 1
    return {s: 100.0 * hits[s] / N_TRIALS for s in STRATEGIES}


def table(title, r_poison, Ms):
    rng = random.Random(SEED)
    print(f"\n================  {title}  (channel trust r_poison={r_poison}) ================")
    print(f"Established: {K_LEGIT} trusted obs of the true value (r={R_LEGIT}). "
          f"Attacker injects M false obs. ASR = % attacks that flip the belief (LOWER=better).")
    hdr = f"{'M (injections)':<16}" + "".join(f"{s:>14}" for s in STRATEGIES)
    print(hdr); print("-" * len(hdr))
    for M in Ms:
        row = f"{M:<16}"
        cell = asr_cell(rng, M, r_poison)
        for s in STRATEGIES:
            row += f"{cell[s]:>13.1f}%"
        print(row)


def crossover(k_values, K_legit):
    """Where does the defense break? ASR for nous_trust as channel trust r_poison rises,
    at several injection volumes. k_values = # competing values (tests the 2-value artifact)."""
    global VALUES, K_LEGIT
    VALUES = [f"v{i}" for i in range(k_values)]
    K_LEGIT = K_legit
    rng = random.Random(SEED)
    Ms = [1, 3, 10, 50]
    print(f"\n----  nous_trust ASR crossover  (k={k_values} values, {K_legit} trusted obs)  ----")
    hdr = f"{'r_poison':<10}" + "".join(f"{'M='+str(m):>10}" for m in Ms)
    print(hdr); print("-" * len(hdr))
    for r in [0.30, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
        row = f"{r:<10}"
        for M in Ms:
            hits = 0
            for _ in range(N_TRIALS):
                V, W = rng.sample(VALUES, 2)
                if predict("nous_trust", [V] * K_LEGIT, [W] * M, r) == W:
                    hits += 1
            row += f"{100.0*hits/N_TRIALS:>9.1f}%"
        print(row)
    VALUES = [f"v{i}" for i in range(4)]; K_LEGIT = 4     # restore


def main():
    Ms = [1, 3, 5, 10, 20, 50]
    # (1) Realistic low-trust compromised channel (e.g. scraped web / user-generated content).
    table("LOW-TRUST CHANNEL  (attacker confined to an untrusted source)", 0.2, Ms)
    # (2) Mid-trust channel — where does the defense start to erode?
    table("MID-TRUST CHANNEL  (partially trusted source)", 0.5, Ms)
    # (3) Worst case / honesty guard: attacker forges FULL trust (no source discrimination).
    table("FORGED HIGH TRUST  (attacker mimics a trusted source -- defense should FAIL)", 0.9, Ms)

    print("\nReading guide:")
    print("  * nous_trust vs majority/lww at low r_poison = the security claim (ASR reduction).")
    print("  * nous_notrust ~ majority at every r_poison = proof the SOURCE-TRUST signal (not the")
    print("    Bayesian math alone) is what defends: strip trust and the posterior is poisonable.")
    print("  * FORGED HIGH TRUST: nous_trust should collapse to nous_notrust -- reliability-")
    print("    weighting cannot defend a payload it (wrongly) trusts. This is the honest boundary.")
    print("  * Watch the injection volume M: the crossover where a low-trust flood still wins is")
    print("    the number that decides whether this is a real defense or a toy.")

    # Crossover: exactly where does trust-weighting break, and is 0% a 2-value artifact?
    crossover(k_values=2, K_legit=4)
    crossover(k_values=4, K_legit=4)
    crossover(k_values=8, K_legit=2)     # many competing values, weakly-held belief (harder)


if __name__ == "__main__":
    main()
