"""LLM-based personal fact extraction for the temporal graph.

Two extraction passes per session:
1. Personal facts: (attribute, value) triples about the user's life
2. Timestamped events: (event, date) pairs crucial for temporal-reasoning questions
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# --- Pass 1: personal facts (original) ---
_FACTS_SYSTEM = """You extract personal facts about the USER from a conversation.
Output ONLY facts the user states about themselves, their life, activities, possessions, or events.
Ignore the assistant's generic advice.

Return a JSON array. Each object:
{"attribute": "short key (2-4 words)", "value": "the specific value", "date": "date if stated"}

Rules:
- Extract SPECIFIC facts: names, places, numbers, dates, titles, brands
- attribute = what kind of fact, value = the exact detail
- Capture: degree earned, where they work/live, purchases, events attended, things created/named, times/durations
- If nothing personal, return []
- Max 12 facts per session"""

_FACTS_PROMPT = """Extract personal facts about the user from this conversation:

{transcript}

Return JSON array:"""

# --- Pass 2: timestamped events (new) ---

# --- Pass 2.5: assistant recommendations ---
_ASST_SYSTEM = """You extract things the ASSISTANT recommended, suggested, or provided in this conversation.
Focus on: named recommendations (restaurants, hotels, places, books, products, resources, people).

Return a JSON array. Each object:
{"category": "type of recommendation (e.g. restaurant, hotel, book)", "name": "the specific name recommended", "context": "brief why or where"}

Rules:
- Only named/specific recommendations, not generic advice
- name = the exact name (e.g. "Roscioli", "International Budget Hostel", "Adobe Premiere Pro tutorials")
- Max 8 recommendations per session
- If no specific named recommendations, return []"""

_ASST_PROMPT = """Extract specific named recommendations the ASSISTANT made in this conversation:

{transcript}

Return JSON array:"""

_EVENTS_SYSTEM = """You extract DATED EVENTS from a conversation — things the user did on a specific date.
Focus on: visits, trips, purchases, meetings, activities, completions, starts.

Return a JSON array. Each object:
{"event": "what happened (short phrase)", "date": "the date (use session date if not explicit)", "duration_days": null_or_number}

Rules:
- Only include events with a clear date (explicit or from session context)
- event = 3-8 word description of what the user did
- date = the date it happened (use the session_date provided if the event happened "today" or "recently")
- duration_days = number if a duration is mentioned (e.g. "3-day trip" → 3), else null
- Max 8 events per session
- If no clear dated events, return []

Examples:
- "I went to MoMA today" → {"event": "visited Museum of Modern Art", "date": "<session_date>", "duration_days": null}
- "I just got back from a 3-day camping trip to Big Sur" → {"event": "camping trip to Big Sur", "date": "<session_date>", "duration_days": 3}"""

_EVENTS_PROMPT = """Session date: {session_date}

Extract dated events the user did from this conversation:

{transcript}

Return JSON array:"""


class LLMFactExtractor:
    """Extracts personal facts + timestamped events via LLM. Async, batched."""

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
        """Extract personal facts + dated events. Returns merged list."""
        if not transcript.strip():
            return []

        text = transcript[:8000]
        own_client = client is None
        if own_client:
            client = httpx.AsyncClient(timeout=45.0)

        try:
            # Run three passes concurrently
            facts_task = self._call(
                client,
                _FACTS_SYSTEM,
                _FACTS_PROMPT.format(transcript=text),
            )
            events_task = self._call(
                client,
                _EVENTS_SYSTEM,
                _EVENTS_PROMPT.format(transcript=text, session_date=session_date or "unknown"),
            )
            asst_task = self._call(
                client,
                _ASST_SYSTEM,
                _ASST_PROMPT.format(transcript=text),
            )
            facts_raw, events_raw, asst_raw = await asyncio.gather(facts_task, events_task, asst_task)

            facts = _parse_json_array(facts_raw)
            events = _parse_json_array(events_raw)
            asst_recs = _parse_json_array(asst_raw)

            # Normalize facts — attach session_date fallback
            result = []
            _VAGUE = {"today", "not specified", "recently", "current", "now",
                      "unknown", "n/a", "none", "yesterday", "earlier", "",
                      "not mentioned", "mentioned", "this week"}
            for f in facts:
                attr = str(f.get("attribute", "")).strip()[:60]
                val = str(f.get("value", "")).strip()[:200]
                if not attr or not val:
                    continue
                fdate = str(f.get("date") or "").strip()
                if fdate.lower() in _VAGUE:
                    fdate = session_date
                result.append({
                    "attribute": attr,
                    "value": val,
                    "date": fdate or session_date,
                    "fact_type": "personal_fact",
                })

            # Normalize events — store as attribute=event_<event>, value=date
            for e in events:
                event_desc = str(e.get("event", "")).strip()[:100]
                edate = str(e.get("date") or "").strip()
                if not event_desc:
                    continue
                if not edate or edate.lower() in _VAGUE:
                    edate = session_date
                # Store event as a fact with attribute=event and value=description
                result.append({
                    "attribute": "event",
                    "value": event_desc,
                    "date": edate or session_date,
                    "fact_type": "dated_event",
                })
                # If duration, store that too
                dur = e.get("duration_days")
                if dur is not None:
                    try:
                        result.append({
                            "attribute": f"duration_{event_desc[:40]}",
                            "value": f"{int(dur)} days",
                            "date": edate or session_date,
                            "fact_type": "event_duration",
                        })
                    except (TypeError, ValueError):
                        pass

            # Normalize assistant recommendations
            for rec in asst_recs:
                cat = str(rec.get("category", "recommendation")).strip()[:40]
                name = str(rec.get("name", "")).strip()[:120]
                if not name:
                    continue
                result.append({
                    "attribute": f"recommended_{cat.lower().replace(' ', '_')}",
                    "value": name,
                    "date": session_date,
                    "fact_type": "assistant_recommendation",
                })

            return result

        except Exception as e:
            logger.debug("LLM fact extraction failed: %s", e)
            return []
        finally:
            if own_client:
                await client.aclose()

    async def _call(self, client: httpx.AsyncClient, system: str, prompt: str) -> str:
        resp = await client.post(
            f"{self._base_url}/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 800,
            },
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _parse_json_array(content: str) -> list[dict[str, Any]]:
    content = content.strip()
    if content.startswith("```"):
        content = "\n".join(
            line for line in content.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [f for f in parsed if isinstance(f, dict)]
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list):
                    return [f for f in parsed if isinstance(f, dict)]
            except json.JSONDecodeError:
                pass
    return []


# Needed for concurrent calls inside extract()
import asyncio  # noqa: E402
