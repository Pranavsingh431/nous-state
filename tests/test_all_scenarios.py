"""
Nous Architecture Tests — Tests the engine API directly.

These tests bypass the extractor entirely and call the engine's internal
update mechanism directly. This decouples architectural correctness from
NLP parsing quality — the right way to unit test.

Run: python tests/test_all_scenarios.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nous.engine import Nous


def inject(nous: Nous, entity: str, attribute: str, value: str,
           source: str = "test", reliability: float = 0.9,
           timestamp: float = None):
    """
    Bypass the extractor and inject a claim directly into the engine.
    Calls the updater directly via world_model + delta_log.
    """
    dim_id = f"{entity}.{attribute}"
    dim = nous.world_model.get_dimension(dim_id)
    delta = nous.updater.update(
        dimension=dim,
        observed_value=value,
        evidence=f"[test] {entity}.{attribute} = {value}",
        source=source,
        reliability=reliability,
        timestamp=timestamp or time.time()
    )
    nous.world_model.save_dimension(dim_id)
    nous.delta_log.append(delta)
    return delta


def run_tests():
    print("Running Nous Architecture Scenarios...\n")

    db_path = "test_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    nous = Nous(db_path)

    # ──────────────────────────────────────────────────────────────────────
    print("--- Scenario 1: Knowledge Conflict ---")

    inject(nous, "Pranav", "employer", "Google", timestamp=1000)
    p_google = nous.query("Pranav", "employer").get("Google", 0.0)
    print(f"After observation 1, P(Google): {p_google:.2f}")
    assert p_google > 0.8, f"Expected > 0.8, got {p_google}"

    inject(nous, "Pranav", "employer", "OpenAI", timestamp=2000)
    p_openai = nous.query("Pranav", "employer").get("OpenAI", 0.0)
    p_google_new = nous.query("Pranav", "employer").get("Google", 0.0)
    print(f"After observation 2, P(OpenAI): {p_openai:.2f}, P(Google): {p_google_new:.2f}")
    assert p_openai > 0.8, f"Expected OpenAI > 0.8, got {p_openai}"
    assert p_google_new < 0.2, f"Expected Google < 0.2, got {p_google_new}"

    # Time-travel
    past_google = nous.query_at("Pranav", "employer", 1500).get("Google", 0.0)
    print(f"Time-travel to timestamp 1500, P(Google): {past_google:.2f}")
    assert past_google > 0.8, f"Expected past Google > 0.8, got {past_google}"

    history = nous.history("Pranav", "employer")
    surprise_2 = history[1].surprise
    print(f"Number of deltas: {len(history)}")
    print(f"Surprise of second observation: {surprise_2:.2f} bits")
    assert len(history) == 2
    assert surprise_2 > 2.0, f"Expected surprise > 2.0, got {surprise_2}"
    print("Scenario 1 PASSED.\n")

    # ──────────────────────────────────────────────────────────────────────
    print("--- Scenario 2: Persistence ---")
    nous.close()

    nous2 = Nous(db_path)
    p_openai_reloaded = nous2.query("Pranav", "employer").get("OpenAI", 0.0)
    print(f"Reloaded memory, P(OpenAI): {p_openai_reloaded:.2f}")
    assert p_openai_reloaded > 0.8, f"Expected reloaded OpenAI > 0.8, got {p_openai_reloaded}"

    nous = nous2
    print("Scenario 2 PASSED.\n")

    # ──────────────────────────────────────────────────────────────────────
    print("--- Scenario 3: Identity Resolution ---")

    inject(nous, "Pranav Singh", "project", "NyayaSahayak")
    inject(nous, "pranav431", "project", "NyayaSahayak")

    coupling = nous.get_coupling("Pranav Singh", "pranav431")
    print(f"Coupling score with one shared attribute: {coupling:.2f}")
    assert coupling == 0.0, f"Expected no coupling from one shared attribute, got {coupling}"

    inject(nous, "Pranav Singh", "employer", "Google DeepMind")
    inject(nous, "pranav431", "employer", "Google DeepMind")
    inject(nous, "Pranav Singh", "role", "ML engineer")
    inject(nous, "pranav431", "role", "ML engineer")

    coupling = nous.get_coupling("Pranav Singh", "pranav431")
    print(f"Coupling score with three shared attributes: {coupling:.2f}")
    assert coupling > 0.5, f"Expected coupling > 0.5 with enough overlap, got {coupling}"
    print("Scenario 3 PASSED.\n")

    # ──────────────────────────────────────────────────────────────────────
    print("--- Scenario 4: Forgetting (Entropy Decay) ---")

    inject(nous, "Alice", "ui_preference", "dark mode", timestamp=3000)
    p_dark = nous.query("Alice", "ui_preference").get("dark mode", 0.0)
    print(f"Initial P(dark mode): {p_dark:.2f}")
    assert p_dark > 0.7, f"Expected initial dark mode > 0.7, got {p_dark}"

    # Simulate time passing (extreme decay rate for test speed)
    nous.compressor.decay_rate = 0.5
    nous.apply_decay(current_time=3000 + 10)

    p_dark_decayed = nous.query("Alice", "ui_preference").get("dark mode", 0.0)
    print(f"After decay, P(dark mode): {p_dark_decayed:.2f}")
    assert p_dark_decayed < p_dark, f"Expected decay: {p_dark_decayed} < {p_dark}"
    print("Scenario 4 PASSED.\n")

    # ──────────────────────────────────────────────────────────────────────
    print("--- Scenario 5: Surprise Calibration ---")

    # Brand new dimension → should get novelty prior (not 0), so surprise < 20 bits
    nous2_clean = Nous(":memory:")
    delta_new = inject(nous2_clean, "Bob", "employer", "Stripe")
    print(f"First observation surprise (new entity): {delta_new.surprise:.2f} bits")
    assert delta_new.surprise < 20.0, "Surprise should be calibrated, not maxed at 20"

    # Reinforcement → very low surprise
    inject(nous2_clean, "Bob", "employer", "Stripe")
    d_reinforce = inject(nous2_clean, "Bob", "employer", "Stripe")
    print(f"Reinforcement surprise: {d_reinforce.surprise:.2f} bits")
    assert d_reinforce.surprise < 1.0, f"Reinforcement should be < 1 bit, got {d_reinforce.surprise}"
    nous2_clean.close()
    print("Scenario 5 PASSED.\n")

    # ──────────────────────────────────────────────────────────────────────
    nous.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    print("ALL SCENARIOS PASSED.")


if __name__ == "__main__":
    run_tests()
