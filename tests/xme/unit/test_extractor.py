"""Unit tests for FactExtractor regex mode."""
import pytest
from xme.extraction.extractor import FactExtractor, _extract_assistant_text, _parse_json_list
from xme.models import FactType


_SAMPLE = """
### 2026-07-15 10:00 UTC
**User**: How should we handle the database?

### 2026-07-15 10:01 UTC
**Assistant**: We decided to use PostgreSQL for persistence because it has better ACID guarantees.

### 2026-07-15 10:05 UTC
**User**: What about the lock approach?

### 2026-07-15 10:06 UTC
**Assistant**: The Redis distributed lock failed because of timeout under high load.
We should prefer an eventually consistent approach.

### 2026-07-15 10:10 UTC
**Assistant**: We always use pytest for testing in this project.
"""


class TestRegexExtractor:
    def setup_method(self):
        self.extractor = FactExtractor(llm_api_key="")  # no LLM — regex only

    @pytest.mark.asyncio
    async def test_extracts_decisions(self):
        facts = await self.extractor.extract(_SAMPLE, "proj", "user")
        decision_facts = [f for f in facts if f.fact_type == FactType.DECISION.value]
        assert len(decision_facts) >= 1
        contents = " ".join(f.content for f in decision_facts).lower()
        assert "postgresql" in contents

    @pytest.mark.asyncio
    async def test_extracts_attempts(self):
        facts = await self.extractor.extract(_SAMPLE, "proj", "user")
        attempt_facts = [f for f in facts if f.fact_type == FactType.ATTEMPT.value]
        assert len(attempt_facts) >= 1
        contents = " ".join(f.content for f in attempt_facts).lower()
        assert "failed" in contents or "timeout" in contents or "redis" in contents

    @pytest.mark.asyncio
    async def test_extracts_preferences(self):
        facts = await self.extractor.extract(_SAMPLE, "proj", "user")
        pref_facts = [f for f in facts if f.fact_type == FactType.PREFERENCE.value]
        assert len(pref_facts) >= 1
        contents = " ".join(f.content for f in pref_facts).lower()
        assert "pytest" in contents

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self):
        facts = await self.extractor.extract("", "proj", "user")
        assert facts == []

    @pytest.mark.asyncio
    async def test_no_duplicates(self):
        facts = await self.extractor.extract(_SAMPLE, "proj", "user")
        titles = [f.title for f in facts]
        # No two facts should have identical titles
        assert len(titles) == len(set(titles))

    @pytest.mark.asyncio
    async def test_project_id_set(self):
        facts = await self.extractor.extract(_SAMPLE, "my-project", "raj")
        assert all(f.project_id == "my-project" for f in facts)

    @pytest.mark.asyncio
    async def test_source_episode_id_set(self):
        facts = await self.extractor.extract(_SAMPLE, "proj", "user", source_episode_id="ep-123")
        assert all(f.source_episode_id == "ep-123" for f in facts)


class TestHelpers:
    def test_extract_assistant_text(self):
        text = "**User**: hello\n**Assistant**: world\n### next\n**User**: q"
        extracted = _extract_assistant_text(text)
        assert "world" in extracted
        assert "hello" not in extracted

    def test_parse_json_list_clean(self):
        raw = '[{"fact_type": "decision", "title": "test"}]'
        result = _parse_json_list(raw)
        assert len(result) == 1
        assert result[0]["title"] == "test"

    def test_parse_json_list_with_code_fence(self):
        raw = '```json\n[{"fact_type": "decision"}]\n```'
        result = _parse_json_list(raw)
        assert len(result) == 1

    def test_parse_json_list_invalid(self):
        assert _parse_json_list("not json") == []
