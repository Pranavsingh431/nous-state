"""
Rule-based claim extractor using regex patterns.
Handles multi-word entities, CamelCase names, and multi-word values.
Used as the zero-dependency fallback when no LLM extractor is provided.
"""
from typing import List, Tuple
import re


# Matches: a capitalized word, a CamelCase word, or an alphanumeric handle
# e.g. "Pranav", "NyayaSahayak", "pranav431", "Pranav Singh", "Google DeepMind"
_ENTITY = r'(?:[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*|[a-z][a-z0-9_]+)'

# Stop words that terminate a value match (temporal, prepositions, conjunctions)
_STOP = r'(?:last|this|next|because|since|for|in|at|on|to|from|and|but|that|which|where|when|who|how|the\s)'

# Matches a value: CamelCase, ALL-CAPS acronym, word with digits/hyphens,
# or a multi-word title-case phrase — but stops at stop-words
_VALUE_WORD = r'(?:[A-Z0-9][a-zA-Z0-9_\-\.]*|[a-z][a-z0-9_\-\.]+)'
# A value is 1 word, optionally followed by more words that are NOT stop-words
_VALUE = rf'{_VALUE_WORD}(?:\s+(?!{_STOP}){_VALUE_WORD}){{0,4}}'


def _c(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# Each rule: (compiled_pattern, attribute_name)
# Groups named `entity` and `value` must be present in each pattern.
RULES: List[Tuple[re.Pattern, str]] = [

    # ── Employment ────────────────────────────────────────────────────────
    # "works at" / "is employed at" — value is proper-noun org name
    (_c(rf'(?P<entity>{_ENTITY})\s+(?:works\s+at|is\s+employed\s+(?:at|by))\s+(?P<value>{_VALUE})'),
     "employer"),
    # "joined X" — stop before temporal words like "last week"
    (_c(rf'(?P<entity>{_ENTITY})\s+joined\s+(?P<value>{_VALUE_WORD}(?:\s+(?!(?:last|this|next|week|month|year|ago|because)){_VALUE_WORD}){{0,3}})'),
     "employer"),

    (_c(rf'(?P<entity>{_ENTITY})\s+(?:left|quit|departed\s+from)\s+(?P<value>{_VALUE})'),
     "former_employer"),

    # ── Role / Title ───────────────────────────────────────────────────────
    (_c(rf'(?P<entity>{_ENTITY})\s+is\s+(?:an?\s+)?(?P<value>(?:[A-Z][a-z]+\s+)*(?:engineer|developer|researcher|scientist|manager|founder|director|lead|analyst|designer|architect))'),
     "role"),

    # ── Location ───────────────────────────────────────────────────────────
    (_c(rf'(?P<entity>{_ENTITY})\s+(?:lives?\s+in|moved?\s+to|is\s+(?:based|located)\s+in|relocated\s+to)\s+(?P<value>{_VALUE})'),
     "location"),

    # ── Technology / Model ─────────────────────────────────────────────────
    # "<Entity> uses <Value>"  or  "<Entity> is using <Value>"
    (_c(rf'(?P<entity>{_ENTITY})\s+(?:uses?|is\s+using|switched?\s+to|migrated?\s+to|now\s+uses?)\s+(?P<value>{_VALUE_WORD}(?:\s+{_VALUE_WORD}){{0,3}})'),
     "model"),

    # "<Entity> is built on / powered by / integrated with <Value>"
    (_c(rf'(?P<entity>{_ENTITY})\s+(?:is\s+)?(?:built\s+on|powered\s+by|integrated\s+with)\s+(?P<value>{_VALUE})'),
     "model"),

    # ── Project / Building ──────────────────────────────────────────────────
    (_c(rf'(?P<entity>{_ENTITY})\s+is\s+building\s+(?P<value>{_ENTITY})'),
     "project"),

    # ── Preference ─────────────────────────────────────────────────────────
    (_c(rf'(?P<entity>{_ENTITY})\s+prefer(?:s|red)?\s+(?P<value>{_VALUE})'),
     "preference"),

    # ── Type / Category ────────────────────────────────────────────────────
    # "NyayaSahayak is a legal AI assistant"
    (_c(rf'(?P<entity>{_ENTITY})\s+is\s+an?\s+(?P<value>(?:[a-z]+\s+){{0,4}}(?:assistant|tool|platform|system|app|service|library|framework|model|bot|agent|api))'),
     "type"),

    # ── Team ───────────────────────────────────────────────────────────────
    (_c(rf'(?P<entity>{_ENTITY})\s+(?:works?\s+on|is\s+on|joined)\s+the\s+(?P<value>{_ENTITY})\s+team'),
     "team"),

    # ── Language / Framework ───────────────────────────────────────────────
    (_c(rf'(?P<entity>{_ENTITY})\s+(?:is\s+)?(?:written\s+in|built\s+(?:with|using)|implemented\s+in)\s+(?P<value>{_VALUE})'),
     "language"),
]


class Extractor:
    """
    Zero-dependency rule-based claim extractor.
    Handles: CamelCase entities, multi-word values, common NLP patterns.
    For production use with messy natural language, use LLMExtractor instead.
    """

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Returns a list of (entity, attribute, value) tuples.
        Deduplicates by (entity, attribute) — last match wins.
        """
        seen: dict = {}  # (entity, attribute) → value

        for pattern, attribute in RULES:
            for match in pattern.finditer(text):
                groups = match.groupdict()
                entity = self._clean(groups.get("entity", ""))
                value = self._clean(groups.get("value", ""))

                if not entity or not value:
                    continue
                if entity.lower() == value.lower():
                    continue  # skip self-referential nonsense

                key = (entity, attribute)
                seen[key] = value

        return [(entity, attr, value) for (entity, attr), value in seen.items()]

    def _clean(self, s: str) -> str:
        """Strip articles, determiners, trailing punctuation, and extra whitespace."""
        s = s.strip()
        # Remove leading articles
        s = re.sub(r'^(?:the|a|an)\s+', '', s, flags=re.IGNORECASE)
        # Strip trailing sentence punctuation
        s = s.rstrip('.,;:!?')
        # Collapse internal whitespace
        s = " ".join(s.split())
        return s
