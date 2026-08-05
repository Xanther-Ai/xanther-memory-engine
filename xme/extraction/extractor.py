"""FactExtractor — extract structured facts from raw text.

Two modes:
  LLM mode  (llm_api_key set): structured extraction via OpenRouter
  Regex mode (no key):         heuristic extraction — good for common patterns
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from xme.models import Confidence, Fact, FactType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_DECISION_RE = [
    re.compile(r"(?:decided?|decision|we(?:'ll| will)? use|chose|going with|switching to)\s*[:\-]?\s*(.{10,250})", re.I),
    re.compile(r"ADR\s*[:\-]\s*(.{10,250})", re.I),
]
_ATTEMPT_RE = [
    re.compile(r"(?:tried?|attempted?|gave up on|failed|didn'?t work|broke)\s*[:\-]?\s*(.{10,250})", re.I),
    re.compile(r"(?:failure reason|because|root cause)\s*[:\-]?\s*(.{10,200})", re.I),
]
_PREFERENCE_RE = [
    re.compile(r"(?:prefer|always use|we use|I use|use only|our convention is)\s+(.{5,150})", re.I),
]
_CONVENTION_RE = [
    re.compile(r"(?:convention|team rule|standard|policy|all PRs?)\s*[:\-]?\s*(.{10,250})", re.I),
    re.compile(r"(?:must|should|never|always)\s+(.{10,150})", re.I),
]
_ENTITY_RE = [
    re.compile(r"(?:component|service|module|system|library|framework|tool|database)\s+[`\"]?(\w[\w\-\.]+)\b", re.I),
]

_LLM_SYSTEM = """\
You are a memory extraction assistant. Extract structured facts from the conversation.
Return a JSON array of objects. Each object:
{
  "fact_type": "decision"|"attempt"|"preference"|"convention"|"entity",
  "title": "short title (max 80 chars)",
  "content": "detailed description",
  "confidence": "EXTRACTED",
  "metadata": {}  // type-specific: decisions have outcome, attempts have failure_reason
}
Only extract clearly stated facts. If nothing clear, return [].
"""


class FactExtractor:
    """Extract Fact objects from raw text content."""

    def __init__(
        self,
        llm_api_key: str = "",
        llm_model: str = "openai/gpt-4o-mini",
        llm_base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self._api_key = llm_api_key
        self._model = llm_model
        self._base_url = llm_base_url

    async def extract(
        self,
        text: str,
        project_id: str,
        user_id: str,
        source_episode_id: Optional[str] = None,
    ) -> list[Fact]:
        """Extract facts from text. Uses LLM if key available, else regex."""
        if not text.strip():
            return []

        if self._api_key:
            try:
                return await self._llm_extract(text, project_id, user_id, source_episode_id)
            except Exception as e:
                logger.warning("LLM extraction failed, falling back to regex: %s", e)

        return self._regex_extract(text, project_id, user_id, source_episode_id)

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    async def _llm_extract(
        self,
        text: str,
        project_id: str,
        user_id: str,
        source_episode_id: Optional[str],
    ) -> list[Fact]:
        import httpx

        # Truncate long texts to stay within context limits
        truncated = text[:6000] if len(text) > 6000 else text

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": f"Extract facts from:\n\n{truncated}"},
            ],
            "temperature": 0.0,
            "max_tokens": 2000,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

        raw_list = _parse_json_list(content)
        facts: list[Fact] = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            fact = Fact(
                fact_type=item.get("fact_type", FactType.ENTITY.value),
                project_id=project_id,
                user_id=user_id,
                title=item.get("title", "")[:100],
                content=item.get("content", ""),
                metadata=item.get("metadata", {}),
                source_episode_id=source_episode_id,
                confidence=Confidence.EXTRACTED.value,
            )
            if fact.title and fact.content:
                facts.append(fact)

        logger.debug("LLM extracted %d facts", len(facts))
        return facts

    # ------------------------------------------------------------------
    # Regex extraction
    # ------------------------------------------------------------------

    def _regex_extract(
        self,
        text: str,
        project_id: str,
        user_id: str,
        source_episode_id: Optional[str],
    ) -> list[Fact]:
        # Only scan assistant turns to avoid false positives from user questions
        assistant_text = _extract_assistant_text(text)

        facts: list[Fact] = []

        def _collect(patterns: list[re.Pattern], fact_type: str) -> None:  # type: ignore[type-arg]
            seen: set[str] = set()
            for pat in patterns:
                for m in pat.finditer(assistant_text):
                    raw = m.group(1).strip().rstrip(".,;:)")
                    norm = raw.lower()[:60]
                    if len(raw) >= 10 and norm not in seen:
                        seen.add(norm)
                        facts.append(Fact(
                            fact_type=fact_type,
                            project_id=project_id,
                            user_id=user_id,
                            title=raw[:100],
                            content=raw,
                            source_episode_id=source_episode_id,
                            confidence=Confidence.EXTRACTED.value,
                        ))

        _collect(_DECISION_RE, FactType.DECISION.value)
        _collect(_ATTEMPT_RE, FactType.ATTEMPT.value)
        _collect(_PREFERENCE_RE, FactType.PREFERENCE.value)
        _collect(_CONVENTION_RE, FactType.CONVENTION.value)
        _collect(_ENTITY_RE, FactType.ENTITY.value)

        logger.debug("Regex extracted %d facts", len(facts))
        return facts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_assistant_text(text: str) -> str:
    """Pull only assistant/AI turns from a markdown transcript."""
    turns = re.findall(r"\*\*Assistant\*\*:\s*(.*?)(?=\n###|\Z)", text, re.DOTALL)
    if turns:
        return "\n".join(turns)
    # No structured format — scan all text
    return text


def _parse_json_list(content: str) -> list:
    """Extract JSON array from LLM response, handling markdown code blocks."""
    content = content.strip()
    # Strip markdown code fences
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "facts" in parsed:
            return parsed["facts"]
    except json.JSONDecodeError:
        # Try extracting first [...] block
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return []
