"""
Query rewriter for Nous retrieval pipeline.
Decomposes complex questions into simpler sub-queries.
Zero new dependencies — uses OpenRouter via urllib.
"""

import json
import urllib.request
import os


class QueryRewriter:
    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model

    def decompose(self, question: str) -> list[str]:
        """
        Decompose a complex question into 1-3 simpler sub-queries.
        Returns [original_question] if decomposition fails or isn't needed.

        Examples:
        "What does Caroline's sister's friend study?"
        → ["Who is Caroline's sister?",
           "Who is [sister]'s friend?",
           "What does [friend] study?"]

        "Where does Pranav work?"
        → ["Where does Pranav work?"]  (already simple, no decomposition)
        """
        prompt = f"""You are a query decomposer for a memory retrieval system.

Given a question, decompose it into 1-3 simpler sub-queries if it
requires chaining multiple facts. If the question is already simple
(single fact lookup), return it unchanged.

Rules:
- Return ONLY a JSON array of strings, no explanation
- Maximum 3 sub-queries
- Each sub-query should be answerable from a single fact
- Preserve entity names exactly
- If question is simple, return array with just the original question

Examples:
Input: "What does Caroline's sister's friend study?"
Output: ["Who is Caroline's sister?", "Who is that person's friend?", "What does that person study?"]

Input: "Where does Pranav work?"
Output: ["Where does Pranav work?"]

Input: "Did Caroline meet her college friend before or after moving to NYC?"
Output: ["When did Caroline move to NYC?", "When did Caroline meet her college friend?"]

Now decompose:
Input: "{question}"
Output:"""

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 200
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"].strip()
                content = content.replace("```json", "").replace("```", "").strip()
                sub_queries = json.loads(content)
                if isinstance(sub_queries, list) and len(sub_queries) > 0:
                    return sub_queries[:3]
        except Exception:
            pass

        return [question]
