"""
LLM-powered claim extraction using OpenRouter API.
v2: Context-aware extraction with value normalization.
"""
import json
import re
import urllib.request
import urllib.error
from typing import List, Tuple, Optional, Dict


class LLMExtractor:
    """
    Extracts (entity, attribute, value) claims from natural language
    using an LLM via OpenRouter's API.
    
    v2 improvements:
    - User context injection (solves "I" → entity mapping)
    - Post-processing normalization (solves "Google DeepMind" vs "Google_DeepMind")
    - Known-entity resolution (maps aliases to canonical names)
    """

    SYSTEM_PROMPT_TEMPLATE = """You are a precise knowledge extraction engine. 
Given a natural language statement, extract ALL factual claims as structured tuples.

Each claim must be: {{"entity": "...", "attribute": "...", "value": "..."}}

{context_block}

Rules:
1. Entity = the subject (person, project, company, tool). Use the most specific proper name.
2. Attribute = the property being described. Use snake_case. Standard attributes:
   - employer, role, team, location (for people)
   - model, framework, language, type, strength, status (for projects/tools)
   - ui_preference, editor, editor_keybindings, coding_style (for preferences)
3. Value = the current state. Use the canonical form (proper capitalization, no underscores for names).
4. If someone SWITCHED from X to Y, extract ONLY the NEW value. The engine handles history.
5. Extract EVERY distinct fact. One sentence may contain multiple claims.
6. NEVER use "user", "he", "she", or "they" as an entity. Always resolve to the proper name.
7. If the speaker is talking about themselves ("I", "my", "me"), use the identity specified above.
8. Attribute names must be consistent: always "employer" not sometimes "company" or "workplace".

Respond ONLY with a valid JSON array. No explanation, no markdown fences, no extra text.

Examples:

Input: "I switched from Mistral to GPT-4 for NyayaSahayak because legal reasoning improved."
Output: [{{"entity": "NyayaSahayak", "attribute": "model", "value": "GPT-4"}}, {{"entity": "NyayaSahayak", "attribute": "strength", "value": "legal reasoning"}}]

Input: "I joined Google last month and work on Gemini in Bangalore."
Output: [{{"entity": "{{user_name}}", "attribute": "employer", "value": "Google"}}, {{"entity": "{{user_name}}", "attribute": "team", "value": "Gemini"}}, {{"entity": "{{user_name}}", "attribute": "location", "value": "Bangalore"}}]

Input: "I prefer dark mode and use Vim keybindings in VS Code."
Output: [{{"entity": "{{user_name}}", "attribute": "ui_preference", "value": "dark mode"}}, {{"entity": "{{user_name}}", "attribute": "editor_keybindings", "value": "Vim"}}, {{"entity": "{{user_name}}", "attribute": "editor", "value": "VS Code"}}]"""

    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash",
                 base_url: str = "https://openrouter.ai/api/v1",
                 user_context: Optional[Dict[str, str]] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        # User context for resolving "I" / "my" / "me"
        self.user_context = user_context or {}
        self.user_name = self.user_context.get("name", "user")
        
        # Known entity aliases → canonical name
        self._aliases: Dict[str, str] = {}
        
        # Build the system prompt with context
        self._system_prompt = self._build_prompt()

    def _build_prompt(self) -> str:
        """Build the system prompt with user context injected."""
        if self.user_context:
            context_lines = ["IMPORTANT CONTEXT about the current user:"]
            for k, v in self.user_context.items():
                context_lines.append(f"- {k}: {v}")
            context_lines.append(f'When the speaker says "I", "my", or "me", the entity is "{self.user_name}".')
            context_block = "\n".join(context_lines)
        else:
            context_block = 'When the speaker says "I", "my", or "me", use "user" as the entity.'
            
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            context_block=context_block,
            user_name=self.user_name
        )

    def register_alias(self, alias: str, canonical: str):
        """Register an entity alias. e.g., register_alias("pranav_ml", "Pranav")"""
        self._aliases[self._normalize_entity(alias)] = canonical

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Sends text to the LLM and parses the response into claim tuples.
        Applies normalization and alias resolution.
        Returns: List of (entity, attribute, value) tuples.
        """
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.0,
            "max_tokens": 1024
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            print(f"[LLMExtractor] HTTP {e.code}: {error_body}")
            return []
        except Exception as e:
            print(f"[LLMExtractor] Request failed: {e}")
            return []

        # Parse LLM response
        try:
            content = body["choices"][0]["message"]["content"].strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:])  # remove first line (```json)
                if content.rstrip().endswith("```"):
                    content = content.rstrip()[:-3]
                content = content.strip()

            claims_raw = json.loads(content)
            claims = []
            for c in claims_raw:
                entity = c.get("entity", "").strip()
                attribute = c.get("attribute", "").strip()
                value = c.get("value", "")
                if isinstance(value, (int, float, bool)):
                    value = str(value)
                value = value.strip()
                
                if entity and attribute and value:
                    # Apply normalization
                    entity = self._normalize_entity(entity)
                    attribute = self._normalize_attribute(attribute)
                    value = self._normalize_value(value)
                    
                    # Apply alias resolution
                    entity = self._resolve_alias(entity)
                    
                    claims.append((entity, attribute, value))
            return claims
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"[LLMExtractor] Parse error: {e}")
            raw = body.get('choices', [{}])
            if raw:
                raw_content = raw[0].get('message', {}).get('content', 'N/A')
                print(f"[LLMExtractor] Raw content: {raw_content[:200]}")
            return []

    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity names for consistency."""
        # Replace underscores with spaces
        entity = entity.replace("_", " ")
        # Title case for proper nouns, but keep acronyms
        parts = entity.split()
        normalized = []
        for part in parts:
            if part.isupper() and len(part) > 1:
                normalized.append(part)  # Keep acronyms like "AI", "ML"
            else:
                normalized.append(part.capitalize() if not part[0].isupper() else part)
        return " ".join(normalized)

    def _normalize_attribute(self, attribute: str) -> str:
        """Normalize attribute names to snake_case."""
        # Replace spaces and hyphens with underscores
        attr = attribute.replace(" ", "_").replace("-", "_")
        # Lowercase
        attr = attr.lower()
        # Collapse multiple underscores
        attr = re.sub(r'_+', '_', attr)
        return attr

    def _normalize_value(self, value: str) -> str:
        """Normalize values for consistent comparison."""
        # Replace underscores with spaces for readability
        value = value.replace("_", " ")
        # Strip extra whitespace
        value = " ".join(value.split())
        return value

    def _resolve_alias(self, entity: str) -> str:
        """Resolve entity aliases to canonical names."""
        normalized = self._normalize_entity(entity)
        # Check direct alias match
        if normalized in self._aliases:
            return self._aliases[normalized]
        # Check case-insensitive
        lower = normalized.lower()
        for alias, canonical in self._aliases.items():
            if alias.lower() == lower:
                return canonical
        return entity
