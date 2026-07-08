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
Given a conversation turn, extract factual claims as structured JSON tuples.

Each claim: {{"entity": "...", "attribute": "...", "value": "..."}}

{context_block}

COMMON ATTRIBUTE VOCABULARY (prefer these when they fit, but do not discard facts that need another concise snake_case attribute):
People: employer, role, occupation, location, identity, relationship_status, feeling, hobby, interest, plan, goal, education, age
Relationships: friend, partner, spouse, family, sibling, parent, child, colleague, mentor, pet, roommate, neighbor
Activities/events: activity, event, achievement, research, travel, purchase, date, duration
Projects/objects: model, framework, language, type, status, strength, topic, preference

Rules:
1. Entity = the subject (person, animal, place, project). Use the EXACT proper name from the text.
2. Attribute = prefer the vocabulary above. If none fit, create a short snake_case attribute rather than dropping the fact.
3. Value = the specific fact. Keep it short but complete enough to answer later questions. Use proper capitalization.
4. Extract relationships: "Caroline's friend Melanie" → {{"entity": "Caroline", "attribute": "friend", "value": "Melanie"}}
5. Extract activities/events: "I went to a yoga class" → {{"entity": "{{user_name}}", "attribute": "activity", "value": "yoga class"}}
6. If the speaker says "I"/"my"/"me", use the identity specified above as entity.
7. NEVER use pronouns as entities. Resolve them to proper names from context.
8. Skip greetings, filler ("Hey!", "How are you?", "Good to see you"). Only extract substantive facts.
9. Preserve temporal clues in values when present: "camping in June", "speech last week", "birthday 10 years ago".

Respond ONLY with a valid JSON array. No markdown, no explanation.

Examples:

Input: "Caroline: I went to a LGBTQ support group yesterday and it was so powerful."
Output: [{{"entity": "Caroline", "attribute": "activity", "value": "LGBTQ support group"}}, {{"entity": "Caroline", "attribute": "feeling", "value": "powerful"}}]

Input: "Melanie: I'm planning a camping trip in June with the kids."
Output: [{{"entity": "Melanie", "attribute": "plan", "value": "camping trip"}}, {{"entity": "Melanie", "attribute": "travel", "value": "camping in June"}}]

Input: "Speaker_A: My friend Sarah just got a new puppy named Buddy."
Output: [{{"entity": "{{user_name}}", "attribute": "friend", "value": "Sarah"}}, {{"entity": "Sarah", "attribute": "pet", "value": "Buddy"}}]

Input: "Caroline: I'm a transgender woman and I've been transitioning for about two years now."
Output: [{{"entity": "Caroline", "attribute": "identity", "value": "transgender woman"}}, {{"entity": "Caroline", "attribute": "transition_duration", "value": "two years"}}]

Input: "Melanie: I love pottery, swimming, and painting — I try to do all of them every week."
Output: [{{"entity": "Melanie", "attribute": "hobby", "value": "pottery"}}, {{"entity": "Melanie", "attribute": "hobby", "value": "swimming"}}, {{"entity": "Melanie", "attribute": "hobby", "value": "painting"}}]

Input: "Caroline: I moved here from Sweden about four years ago, so I'm still getting used to the culture."
Output: [{{"entity": "Caroline", "attribute": "origin", "value": "Sweden"}}, {{"entity": "Caroline", "attribute": "location_history", "value": "moved from Sweden 4 years ago"}}]

Input: "Melanie: I'm single and loving it — been focusing on my kids and my art."
Output: [{{"entity": "Melanie", "attribute": "relationship_status", "value": "single"}}]

CRITICAL: Return at most 10 claims. Output MUST be a complete, valid JSON array — never truncate."""

    # --- Reliability-aware extraction (experimental) -----------------------------------------
    # Motivation: the Bayesian belief updater weights each observation by a `reliability` scalar,
    # but the base extractor emits no confidence, so observe() feeds a single fixed reliability to
    # every claim. With constant reliability the posterior degenerates to a soft recency-follower
    # (the contradiction micro-benchmark showed it then ties/loses to last-write-wins). This
    # variant asks the LLM to estimate a per-claim reliability from EPISTEMIC MARKERS in the
    # utterance — how confidently the speaker states the fact — WITHOUT any access to ground truth.
    RELIABILITY_PROMPT_TEMPLATE = """You are a precise knowledge extraction engine.
Given a conversation turn, extract factual claims as structured JSON tuples, and for EACH claim
estimate how RELIABLE the statement is, judged only from how the speaker phrases it.

Each claim: {{"entity": "...", "attribute": "...", "value": "...", "reliability": 0.0-1.0}}

{context_block}

COMMON ATTRIBUTE VOCABULARY (prefer these when they fit, but do not discard facts that need another concise snake_case attribute):
People: employer, role, occupation, location, identity, relationship_status, feeling, hobby, interest, plan, goal, education, age
Relationships: friend, partner, spouse, family, sibling, parent, child, colleague, mentor, pet, roommate, neighbor
Activities/events: activity, event, achievement, research, travel, purchase, date, duration
Projects/objects: model, framework, language, type, status, strength, topic, preference

RELIABILITY RUBRIC — judge ONLY the speaker's epistemic stance (their phrasing), NOT whether you
think the fact is true. Do not use world knowledge to second-guess the value; only the wording.
  0.90-1.00  Explicit correction/confirmation or emphatic certainty:
             "actually it's X", "to be clear, X", "definitely", "for sure", "confirmed", "I now X".
  0.70-0.85  Plain first-hand assertion, no hedging: "I work at X", "my dog is named X".
  0.45-0.65  Mild hedge: "I think X", "pretty sure it's X", "I believe X".
  0.25-0.45  Strong hedge / guess: "maybe X", "might be X", "not totally sure but X", "probably X".
  0.05-0.25  Rumor / secondhand / very unsure: "I heard it could be X", "someone said X, no idea".

Rules:
1. Entity = the subject. Use the EXACT proper name. Resolve "I"/"my"/"me" per the context above.
2. Attribute = prefer the vocabulary above; else a short snake_case attribute.
3. Value = the specific fact, short but complete. Proper capitalization.
4. reliability = a number in [0,1] from the rubric, based purely on phrasing/epistemic markers.
5. Skip greetings and filler. Only extract substantive facts. NEVER use pronouns as entities.

Respond ONLY with a valid JSON array. No markdown, no explanation.

Examples:

Input: "Caroline: I definitely work at Google now — I started on Monday, it's official."
Output: [{{"entity": "Caroline", "attribute": "employer", "value": "Google", "reliability": 0.95}}]

Input: "Melanie: I think I might be moving to Boston? Not totally sure yet honestly."
Output: [{{"entity": "Melanie", "attribute": "location", "value": "Boston", "reliability": 0.3}}]

Input: "Caroline: Actually, forget what I said before — my role is Staff Engineer, confirmed today."
Output: [{{"entity": "Caroline", "attribute": "role", "value": "Staff Engineer", "reliability": 0.95}}]

Input: "Melanie: I heard Sarah maybe got a new job somewhere, could be wrong though."
Output: [{{"entity": "Sarah", "attribute": "employer", "value": "new job", "reliability": 0.15}}]

Input: "Caroline: My partner's name is Alex and we've been together three years."
Output: [{{"entity": "Caroline", "attribute": "partner", "value": "Alex", "reliability": 0.8}}]

CRITICAL: Return at most 10 claims. Output MUST be a complete, valid JSON array — never truncate."""

    QUESTION_PROMPT_TEMPLATE = """You are a precise knowledge extraction engine.
Given a natural language question about a user, extract the entities and attributes being asked about.

Each extraction must be: {{"entity": "...", "attribute": "..."}}

{context_block}

Rules:
1. Entity = the subject (person, project, company, tool). Use the most specific proper name.
2. Attribute = the property being asked about. Use snake_case. Standard attributes:
   - employer, role, team, location, framework, language, model, etc.
3. NEVER use "user", "he", "she", or "they" as an entity. Always resolve to the proper name.
4. If the question asks about the speaker ("I", "my", "me"), use the identity specified above.

Respond ONLY with a valid JSON array. No explanation.

Examples:
Input: "Where do I work now?"
Output: [{{"entity": "{user_name}", "attribute": "employer"}}]

Input: "What framework is NyayaSahayak using?"
Output: [{{"entity": "NyayaSahayak", "attribute": "framework"}}]
"""

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

    def _build_reliability_prompt(self) -> str:
        """System prompt for reliability-aware extraction (same context injection as _build_prompt)."""
        if self.user_context:
            context_lines = ["IMPORTANT CONTEXT about the current user:"]
            for k, v in self.user_context.items():
                context_lines.append(f"- {k}: {v}")
            context_lines.append(f'When the speaker says "I", "my", or "me", the entity is "{self.user_name}".')
            context_block = "\n".join(context_lines)
        else:
            context_block = 'When the speaker says "I", "my", or "me", use "user" as the entity.'
        return self.RELIABILITY_PROMPT_TEMPLATE.format(
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
            "max_tokens": 4096
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

        import time
        body = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8") if e.fp else ""
                if e.code == 429:
                    wait = 15 * (attempt + 1)
                    print(f"[LLMExtractor] HTTP 429 rate limit. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                print(f"[LLMExtractor] HTTP {e.code}: {error_body[:200]}")
                return []
            except Exception as e:
                if attempt < 3:
                    time.sleep(5)
                    continue
                print(f"[LLMExtractor] Request failed: {e}")
                return []

            # Body received — check for API-level errors (429/504 returned as HTTP 200)
            if "error" in body:
                err_code = body["error"].get("code")
                err_msg  = body["error"].get("message", "unknown")
                if err_code in (429, 504, "429", "504"):
                    wait = 15 * (attempt + 1)
                    print(f"[LLMExtractor] API error {err_code} ({err_msg}). Waiting {wait}s...")
                    time.sleep(wait)
                    body = None
                    continue
                print(f"[LLMExtractor] API error {err_code}: {err_msg}")
                return []
            break  # success

        if body is None:
            return []

        try:
            content = body["choices"][0]["message"]["content"].strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:])  # remove first line (```json)
                if content.rstrip().endswith("```"):
                    content = content.rstrip()[:-3]
                content = content.strip()

            claims_raw = self._parse_json_robust(content)
            if claims_raw is None:
                return []
            claims = []
            for c in claims_raw:
                entity = str(c.get("entity") or "").strip()
                attribute = str(c.get("attribute") or "").strip()
                value = c.get("value")
                if value is None:
                    value = ""
                elif isinstance(value, (int, float, bool)):
                    value = str(value)
                value = str(value).strip()
                
                if entity and attribute and value:
                    # Apply normalization
                    entity = self._normalize_entity(entity)
                    attribute = self._normalize_attribute(attribute)
                    value = self._normalize_value(value)
                    
                    # Apply alias resolution
                    entity = self._resolve_alias(entity)
                    
                    claims.append((entity, attribute, value))
            return claims
        except (KeyError, IndexError) as e:
            print(f"[LLMExtractor] Response structure error: {e}")
            print(f"[LLMExtractor] Full API response: {json.dumps(body)[:400]}")
            return []

    def _chat(self, system_prompt: str, user_text: str, max_tokens: int = 4096) -> Optional[list]:
        """Shared LLM call → parsed JSON list (or None). Mirrors extract()'s retry/error handling.
        Kept separate from extract() so the battle-tested eval path is untouched."""
        import time
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        body = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8") if e.fp else ""
                if e.code == 429:
                    time.sleep(15 * (attempt + 1)); continue
                print(f"[LLMExtractor] HTTP {e.code}: {error_body[:200]}")
                return None
            except Exception as e:
                if attempt < 3:
                    time.sleep(5); continue
                print(f"[LLMExtractor] Request failed: {e}")
                return None
            if "error" in body:
                err_code = body["error"].get("code")
                if err_code in (429, 504, "429", "504"):
                    time.sleep(15 * (attempt + 1)); body = None; continue
                print(f"[LLMExtractor] API error {err_code}: {body['error'].get('message', '')}")
                return None
            break
        if body is None:
            return None
        try:
            content = body["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:])
                if content.rstrip().endswith("```"):
                    content = content.rstrip()[:-3]
                content = content.strip()
            return self._parse_json_robust(content)
        except (KeyError, IndexError) as e:
            print(f"[LLMExtractor] Response structure error: {e}")
            return None

    def extract_with_reliability(self, text: str, default_reliability: float = 0.7
                                 ) -> List[Tuple[str, str, str, float]]:
        """Reliability-aware extraction. Returns (entity, attribute, value, reliability) 4-tuples.

        `reliability` is estimated by the LLM from epistemic markers in the utterance (hedging,
        certainty, self-correction) — never from ground truth. Claims missing/invalid reliability
        fall back to `default_reliability`. Same normalization/alias resolution as extract()."""
        claims_raw = self._chat(self._build_reliability_prompt(), text)
        if not claims_raw:
            return []
        claims: List[Tuple[str, str, str, float]] = []
        for c in claims_raw:
            if not isinstance(c, dict):
                continue
            entity = str(c.get("entity") or "").strip()
            attribute = str(c.get("attribute") or "").strip()
            value = c.get("value")
            value = "" if value is None else str(value).strip()
            if not (entity and attribute and value):
                continue
            try:
                rel = float(c.get("reliability", default_reliability))
            except (TypeError, ValueError):
                rel = default_reliability
            rel = min(1.0, max(0.0, rel))
            entity = self._resolve_alias(self._normalize_entity(entity))
            attribute = self._normalize_attribute(attribute)
            value = self._normalize_value(value)
            claims.append((entity, attribute, value, rel))
        return claims

    def _parse_json_robust(self, content: str):
        """
        Try to parse JSON. If it fails (truncated output), recover the
        last complete object in the array and close the array manually.
        Returns a list (possibly empty) or None on unrecoverable failure.
        """
        # First try: direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Second try: find last complete JSON object and close the array
        # Look for last `}` followed only by whitespace/partial text
        last_brace = content.rfind("}")
        if last_brace != -1:
            truncated = content[:last_brace + 1] + "]"
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass

        # Third try: empty array if content is just `[` or similar
        return []

    def parse_question(self, text: str) -> List[Tuple[str, str]]:
        """
        Sends a question to the LLM and parses the response into (entity, attribute) pairs.
        Returns: List of (entity, attribute) tuples.
        """
        if self.user_context:
            context_block = 'When the speaker says "I", "my", or "me", the entity is "{}".'.format(self.user_name)
        else:
            context_block = 'When the speaker says "I", "my", or "me", use "user" as the entity.'
            
        system_prompt = self.QUESTION_PROMPT_TEMPLATE.format(
            context_block=context_block,
            user_name=self.user_name
        )

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
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

        import time
        body = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 15 * (attempt + 1)
                    time.sleep(wait)
                    continue
                return []
            except Exception as e:
                if attempt < 3:
                    time.sleep(5)
                    continue
                return []

            if "error" in body:
                err_code = body["error"].get("code")
                if err_code in (429, 504, "429", "504"):
                    wait = 15 * (attempt + 1)
                    time.sleep(wait)
                    body = None
                    continue
                return []
            break

        if body is None:
            return []

        try:
            content = body["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:])
                if content.rstrip().endswith("```"):
                    content = content.rstrip()[:-3]
                content = content.strip()

            queries_raw = self._parse_json_robust(content)
            if queries_raw is None:
                return []
            queries = []
            for c in queries_raw:
                entity = str(c.get("entity") or "").strip()
                attribute = str(c.get("attribute") or "").strip()
                
                if entity and attribute:
                    entity = self._normalize_entity(entity)
                    attribute = self._normalize_attribute(attribute)
                    entity = self._resolve_alias(entity)
                    queries.append((entity, attribute))
            return queries
        except Exception as e:
            print(f"[LLMExtractor] Parse error in parse_question: {e}")
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
