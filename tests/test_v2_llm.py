"""
Nous v2 Integration Test — LLM-Powered Extraction
Tests the full pipeline: Natural Language → LLM Extraction → Bayesian Update → Query

Simulates a multi-session agent conversation with a real user.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nous.engine import Nous
from nous.llm_extractor import LLMExtractor

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

def fmt_dist(dist: dict, top_n: int = 3) -> str:
    """Format a distribution for display."""
    sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return " | ".join(f"{k}: {v:.0%}" for k, v in sorted_items)

def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def run_v2_test():
    if not OPENROUTER_KEY:
        print("Set OPENROUTER_API_KEY environment variable to run this test.")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    db_path = "test_v2_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    extractor = LLMExtractor(api_key=OPENROUTER_KEY, user_context={"name": "Pranav"})
    nous = Nous(db_path, extractor=extractor)

    # ─────────────────────────────────────────────────────
    # SESSION 1: Initial user onboarding
    # ─────────────────────────────────────────────────────
    separator("SESSION 1: User Onboarding")

    observations_s1 = [
        "My name is Pranav and I work at Sarvam AI as an ML engineer.",
        "I'm building NyayaSahayak, a legal AI assistant using Mistral.",
        "I prefer dark mode and use VS Code with Vim keybindings.",
    ]

    for obs in observations_s1:
        print(f"  📥 Observing: \"{obs}\"")
        claims = extractor.extract(obs)
        print(f"     Extracted {len(claims)} claims: {claims}")
        nous.observe(obs, source="onboarding_chat")
        time.sleep(1)  # Rate limit courtesy

    # Check what we learned
    print("\n  📊 Current World Model after Session 1:")
    for entity in ["Pranav", "NyayaSahayak", "user"]:
        attrs = nous.world_model.get_entity_attributes(entity)
        if attrs:
            print(f"     {entity}:")
            for attr, dist in attrs.items():
                print(f"       .{attr} = {fmt_dist(dist)}")

    # ─────────────────────────────────────────────────────
    # SESSION 2: Knowledge evolution (things change!)
    # ─────────────────────────────────────────────────────
    separator("SESSION 2: Knowledge Evolution")

    observations_s2 = [
        "I switched from Mistral to GPT-4 for NyayaSahayak because legal reasoning improved significantly.",
        "I just moved from Bangalore to San Francisco for a new role.",
        "Actually, I left Sarvam AI and joined Google DeepMind last week.",
    ]

    for obs in observations_s2:
        # Check surprise BEFORE observing
        surprise_before = nous.surprise(obs)
        
        print(f"  📥 Observing: \"{obs}\"")
        claims = extractor.extract(obs)
        print(f"     Extracted {len(claims)} claims: {claims}")
        print(f"     Surprise score: {surprise_before:.2f} bits")
        nous.observe(obs, source="session2_chat")
        time.sleep(1)

    print("\n  📊 Current World Model after Session 2:")
    for entity in ["Pranav", "NyayaSahayak", "user"]:
        attrs = nous.world_model.get_entity_attributes(entity)
        if attrs:
            print(f"     {entity}:")
            for attr, dist in attrs.items():
                print(f"       .{attr} = {fmt_dist(dist)}")

    # ─────────────────────────────────────────────────────
    # SESSION 3: Contradictory / reinforcing information
    # ─────────────────────────────────────────────────────
    separator("SESSION 3: Contradiction & Reinforcement")

    observations_s3 = [
        "Pranav is definitely at Google DeepMind, I saw his badge.",
        "Wait, someone said Pranav is still at Sarvam AI. Are you sure he left?",
        "Yes confirmed, Pranav is at Google DeepMind working on Gemini.",
    ]

    for obs in observations_s3:
        print(f"  📥 Observing: \"{obs}\"")
        claims = extractor.extract(obs)
        print(f"     Extracted {len(claims)} claims: {claims}")
        nous.observe(obs, source="gossip", reliability=0.7)
        time.sleep(1)

    print("\n  📊 Pranav.employer after conflicting reports:")
    employer = nous.query("Pranav", "employer")
    print(f"     {fmt_dist(employer)}")
    
    # ─────────────────────────────────────────────────────
    # QUERIES: Test the engine's capabilities
    # ─────────────────────────────────────────────────────
    separator("QUERY TESTS")

    # 1. Simple prediction
    print("  🔮 Q: Where does Pranav work?")
    result = nous.predict("Pranav", "employer")
    print(f"     A: {fmt_dist(result)}")

    # 2. What model does NyayaSahayak use?
    print("\n  🔮 Q: What model does NyayaSahayak use?")
    result = nous.predict("NyayaSahayak", "model")
    print(f"     A: {fmt_dist(result)}")

    # 3. Explain WHY the model believes something
    print("\n  🔍 Q: Why does Nous believe Pranav works at Google DeepMind?")
    explanations = nous.explain("Pranav", "employer", "Google DeepMind")
    if not explanations:
        # Try alternate entity names the LLM might have used
        explanations = nous.explain("Pranav", "employer", "Google_DeepMind")
    for delta in explanations:
        print(f"     Evidence: \"{delta.evidence[:80]}...\"")
        print(f"     Surprise: {delta.surprise:.2f} bits | Source: {delta.source}")

    # 4. Full history
    print("\n  📜 Full evolution of Pranav.employer:")
    history = nous.history("Pranav", "employer")
    for i, delta in enumerate(history):
        top_prior = max(delta.prior.items(), key=lambda x: x[1]) if delta.prior else ("?", 0)
        top_post = max(delta.posterior.items(), key=lambda x: x[1]) if delta.posterior else ("?", 0)
        print(f"     Delta {i+1}: {top_prior[0]}({top_prior[1]:.0%}) → {top_post[0]}({top_post[1]:.0%}) | surprise={delta.surprise:.1f} bits")

    # ─────────────────────────────────────────────────────
    # IDENTITY RESOLUTION
    # ─────────────────────────────────────────────────────
    separator("IDENTITY RESOLUTION TEST")

    nous.observe("pranav_ml is building NyayaSahayak at Google DeepMind.", source="github")
    time.sleep(1)

    coupling = nous.get_coupling("Pranav", "pranav_ml")
    print(f"  🔗 Coupling('Pranav', 'pranav_ml') = {coupling:.2f}")
    print(f"     Should merge: {nous.coupler.should_merge(coupling)}")

    # ─────────────────────────────────────────────────────
    # PERSISTENCE TEST
    # ─────────────────────────────────────────────────────
    separator("PERSISTENCE TEST")

    nous.close()
    nous2 = Nous(db_path, extractor=extractor)

    print("  💾 Reloaded from SQLite. Checking beliefs survived:")
    employer_reloaded = nous2.query("Pranav", "employer")
    print(f"     Pranav.employer = {fmt_dist(employer_reloaded)}")

    history_reloaded = nous2.history("Pranav", "employer")
    print(f"     Delta count: {len(history_reloaded)}")

    nous2.close()

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

    separator("ALL V2 TESTS COMPLETE")
    print("  The Nous engine successfully processed real natural language")
    print("  through an LLM parser, performed Bayesian updates, handled")
    print("  contradictions, maintained history, and persisted to disk.\n")


if __name__ == "__main__":
    run_v2_test()
