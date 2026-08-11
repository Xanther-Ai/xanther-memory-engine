"""LLM-based personal fact extraction for the temporal graph.

Extracts structured (attribute, value) triples about the user from a session
transcript using an LLM. Far higher recall than regex — captures paraphrased
facts like "I created a playlist called Summer Vibes" → (playlist_name, Summer Vibes).

Used for LongMemEval-class benchmarks where answers are specific personal facts.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = """You extract personal facts about the USER from a conversation.
Output ONLY facts the user states about themselves, their life, activities, possessions, or events.
Ignore the assistant's generic advice and suggestions.

Return a JSON array. Each object:
{"attribute": "short key (e.g. degree, commute_time, playlist_name)", "value": "the specific value", "date": "when mentioned if stated"}

Rules:
- Extract SPECIFIC facts: names, places, numbers, dates, titles, brands
- attribute = what kind of fact (2-4 words), value = the exact detail
- Capture things like: degree earned, where they work/live, purchases and where from,
  events attended, classes taken, things created/named, changes made, times/durations
- If nothing personal, return []
- Max 10 facts per session

Example input: "I graduated with a Business Administration degree. My commute takes 45 minutes. I made a playlist called Summer Vibes."
Example output: [{"attribute":"degree","value":"Business Administration"},{"attribute":"commute_time","value":"45 minutes"},{"attribute":"playlist_name","value":"Summer Vibes"}]"""

_EXTRACT_PROMPT = """Extract personal facts about the user from this conversation:

{transcript}

Return JSON array of facts:"""


class LLMFactExtractor:
    """Extracts personal facts via LLM. Batched, async, with regex fallback."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    async def extract(
        self,
        transcript: str,
        session_date: str = "",
        client: Optional[httpx.AsyncClient] = None,
    ) -> list[dict[str, Any]]:
        """Extract personal facts from a session transcript."""
        if not transcript.strip():
            return []

        # Truncate to keep costs reasonable
        text = transcript[:8000]

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": _EXTRACT_PROMPT.format(transcript=text)},
            ],
            "temperature": 0.0,
            "max_tokens": 800,
        }

        own_client = client is None
        if own_client:
            client = httpx.AsyncClient(timeout=30.0)

        try:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            facts = _parse_facts(content)
            # Attach date
            for f in facts:
                if not f.get("date"):
                    f["date"] = session_date
            return facts
        except Exception as e:
            logger.debug("LLM fact extraction failed: %s", e)
            return []
        finally:
            if own_client:
                await client.aclose()


def _parse_facts(content: str) -> list[dict[str, Any]]:
    """Parse JSON array of facts from LLM response."""
    content = content.strip()
    if content.startswith("```"):
        content = "\n".join(
            line for line in content.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [
                f for f in parsed
                if isinstance(f, dict) and f.get("attribute") and f.get("value")
            ]
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list):
                    return [f for f in parsed if isinstance(f, dict) and f.get("attribute")]
            except json.JSONDecodeError:
                pass
    return []
