"""Unit tests for XME domain models."""
import pytest
from xme.models import (
    Turn, Episode, Fact, FactType, WorkingContext,
    SessionContext, SearchResults, MemorySearchResult, UpsertResult,
    Confidence, Outcome,
)


class TestTurn:
    def test_to_dict_round_trip(self):
        t = Turn(role="user", content="Hello", tool_name=None)
        d = t.to_dict()
        t2 = Turn.from_dict(d)
        assert t2.role == t.role
        assert t2.content == t.content

    def test_as_text_user(self):
        assert "User:" in Turn(role="user", content="Q").as_text()

    def test_as_text_assistant(self):
        assert "Assistant:" in Turn(role="assistant", content="A").as_text()

    def test_as_text_tool(self):
        t = Turn(role="tool", content="result", tool_name="xce_search")
        assert "xce_search" in t.as_text()


class TestEpisode:
    def test_full_transcript(self):
        ep = Episode(project_id="p", user_id="u")
        ep.turns = [
            Turn(role="user", content="What is auth?"),
            Turn(role="assistant", content="Auth uses JWT"),
        ]
        assert "What is auth?" in ep.full_transcript
        assert "JWT" in ep.full_transcript

    def test_turn_count(self):
        ep = Episode()
        ep.turns = [Turn("user", "q"), Turn("assistant", "a")]
        assert ep.turn_count == 2

    def test_to_dict_round_trip(self):
        ep = Episode(project_id="proj", user_id="user1", summary="test")
        ep.turns = [Turn("user", "hello")]
        d = ep.to_dict()
        ep2 = Episode.from_dict(d)
        assert ep2.project_id == "proj"
        assert ep2.summary == "test"
        assert len(ep2.turns) == 1

    def test_default_outcome(self):
        ep = Episode()
        assert ep.outcome == Outcome.UNKNOWN.value


class TestFact:
    def test_to_dict_round_trip(self):
        f = Fact(
            fact_type=FactType.DECISION.value,
            project_id="proj",
            title="Use FastAPI",
            content="We decided FastAPI over Flask",
        )
        d = f.to_dict()
        f2 = Fact.from_dict(d)
        assert f2.title == f.title
        assert f2.fact_type == f.fact_type

    def test_text_for_embedding(self):
        f = Fact(fact_type="decision", title="Use Redis", content="For caching")
        text = f.text_for_embedding()
        assert "decision" in text
        assert "Redis" in text

    def test_default_confidence(self):
        f = Fact()
        assert f.confidence == Confidence.EXTRACTED.value

    def test_default_status(self):
        f = Fact()
        assert f.status == "active"


class TestWorkingContext:
    def test_to_dict_round_trip(self):
        ctx = WorkingContext(
            project_id="p", user_id="u",
            current_task="refactor auth",
            next_steps="deploy to staging",
        )
        d = ctx.to_dict()
        ctx2 = WorkingContext.from_dict(d)
        assert ctx2.current_task == "refactor auth"
        assert ctx2.next_steps == "deploy to staging"

    def test_as_prompt_block_empty(self):
        ctx = WorkingContext(project_id="p", user_id="u")
        block = ctx.as_prompt_block()
        assert "XME Context" in block

    def test_as_prompt_block_with_data(self):
        ctx = WorkingContext(
            project_id="proj", user_id="raj",
            current_task="Fix auth bug",
            recent_decisions=["Use FastAPI", "Use PostgreSQL"],
            next_steps="Deploy after testing",
        )
        block = ctx.as_prompt_block()
        assert "Fix auth bug" in block
        assert "FastAPI" in block
        assert "Deploy after testing" in block


class TestSearchResults:
    def test_all_results_sorted_by_score(self):
        results = SearchResults(query="auth", project_id="p")
        results.facts = [
            MemorySearchResult("facts", "id1", 0.9, "High relevance"),
            MemorySearchResult("facts", "id2", 0.5, "Low relevance"),
        ]
        results.episodic = [
            MemorySearchResult("episodic", "id3", 0.7, "Medium"),
        ]
        all_r = results.all_results
        assert all_r[0].score == 0.9
        assert all_r[1].score == 0.7
        assert all_r[2].score == 0.5


class TestUpsertResult:
    def test_created(self):
        r = UpsertResult(action="created", fact_id="abc")
        assert r.action == "created"
        assert r.merged_with is None

    def test_merged(self):
        r = UpsertResult(action="merged", fact_id="abc", merged_with="old-id", similarity=0.92)
        assert r.action == "merged"
        assert r.similarity == 0.92
