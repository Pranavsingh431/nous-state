"""
Tests for the rule-based Extractor.
Validates NLP pattern coverage: CamelCase entities, multi-word values, all attribute types.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nous.extractor import Extractor

e = Extractor()


def check(text: str, expected_entity: str, expected_attr: str, expected_value: str):
    results = e.extract(text)
    matches = [(ent, attr, val) for ent, attr, val in results
               if ent == expected_entity and attr == expected_attr]
    assert matches, f"FAIL: No ({expected_entity}, {expected_attr}) in {results}\n  Input: {text!r}"
    val = matches[0][2]
    assert val.lower() == expected_value.lower(), \
        f"FAIL: Expected value '{expected_value}', got '{val}'\n  Input: {text!r}"
    print(f"  PASS: ({expected_entity}, {expected_attr}, {val})")


def run_extractor_tests():
    print("--- Extractor Unit Tests ---\n")

    # Employment
    check("Pranav works at Sarvam AI.", "Pranav", "employer", "Sarvam AI")
    check("Pranav works at Google DeepMind.", "Pranav", "employer", "Google DeepMind")
    check("Pranav joined Google last week.", "Pranav", "employer", "Google")

    # Technology / Model
    check("NyayaSahayak uses GPT-4.", "NyayaSahayak", "model", "GPT-4")
    check("NyayaSahayak switched to Gemini.", "NyayaSahayak", "model", "Gemini")
    check("Alice is using VS Code.", "Alice", "model", "VS Code")

    # Location
    check("Pranav moved to San Francisco.", "Pranav", "location", "San Francisco")
    check("Bob is based in Bangalore.", "Bob", "location", "Bangalore")

    # Preference
    check("Alice prefers dark mode.", "Alice", "preference", "dark mode")

    # Project / Building
    check("Pranav is building NyayaSahayak.", "Pranav", "project", "NyayaSahayak")

    # Type
    check("NyayaSahayak is a legal AI assistant.", "NyayaSahayak", "type", "legal AI assistant")

    print("\nAll extractor tests PASSED.\n")


def test_extractor_patterns():
    run_extractor_tests()


if __name__ == "__main__":
    run_extractor_tests()
