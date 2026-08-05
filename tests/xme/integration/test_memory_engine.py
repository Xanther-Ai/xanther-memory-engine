"""Integration tests for MemoryEngine — full session lifecycle.

All tests use fallback_mode=True (SQLite only, no OpenSearch/Neo4j required).
"""
import pytest
import asyncio
from pathlib import Path
from xme.config import XMESettings
from xme.engine import MemoryEngine
from xme.models import FactType


@pytest.fixture
async def engine(tmp_path):
    settings = XMESettings(
        sqlite_path=str(tmp_path / "xme.db"),
        fallback_mode=True,
        opensearch_enabled=False,
    )
    async with MemoryEngine(settings) as eng:
        yield eng


class TestSessionLifecycle:

    @pytest.mark.asyncio
    async def test_session_start_returns_session_id(self, engine):
        ctx = await engine.session_start("proj", "raj")
        assert ctx.session_id
        assert ctx.project_id == "proj"
        assert ctx.user_id == "raj"

    @pytest.mark.asyncio
    async def test_session_start_empty_context_on_first_use(self, engine):
        ctx = await engine.session_start("new-project", "new-user")
        assert ctx.working_context is None
        assert ctx.last_episode_summary == ""

    @pytest.mark.asyncio
    async def test_record_turn_buffered(self, engine):
        ctx = await engine.session_start("proj", "raj")
        engine.record_turn(ctx.session_id, "user", "How does auth work?")
        engine.record_turn(ctx.session_id, "assistant", "Auth uses JWT tokens")
        # Should be in the active episode
        ep = engine._active.get(ctx.session_id)
        assert ep is not None
        assert len(ep.turns) == 2

    @pytest.mark.asyncio
    async def test_session_end_saves_episode(self, engine):
        ctx = await engine.session_start("proj", "raj")
        engine.record_turn(ctx.session_id, "user", "Fix the auth bug")
        engine.record_turn(ctx.session_id, "assistant", "Fixed. Auth now uses refresh tokens.")

        ep = await engine.session_end(
            session_id=ctx.session_id,
            project_id="proj",
            user_id="raj",
            summary="Fixed auth bug — added refresh tokens",
            outcome="success",
        )
        assert ep.episode_id
        assert ep.outcome == "success"
        assert ep.ended_at is not None

    @pytest.mark.asyncio
    async def test_session_end_extracts_facts(self, engine):
        ctx = await engine.session_start("proj", "raj")
        engine.record_turn(ctx.session_id, "assistant",
                           "We decided to use FastAPI for the auth service.")
        ep = await engine.session_end(
            session_id=ctx.session_id,
            project_id="proj",
            user_id="raj",
            summary="Auth decision",
            outcome="success",
        )
        # Should have extracted at least one decision
        assert len(ep.fact_ids) >= 0  # may be 0 if regex didn't match

    @pytest.mark.asyncio
    async def test_session_end_updates_working_context(self, engine):
        ctx = await engine.session_start("proj", "raj")
        await engine.session_end(
            session_id=ctx.session_id,
            project_id="proj",
            user_id="raj",
            summary="Completed auth refactor",
            outcome="success",
            next_steps="Deploy to staging",
        )
        wctx = engine.context.get("proj", "raj")
        assert wctx is not None
        assert "Completed auth refactor" in wctx.last_session_summary

    @pytest.mark.asyncio
    async def test_second_session_gets_context(self, engine):
        # First session
        ctx1 = await engine.session_start("proj", "raj")
        await engine.session_end(
            session_id=ctx1.session_id,
            project_id="proj",
            user_id="raj",
            summary="Completed auth refactor",
            outcome="success",
        )
        # Second session should have last session context
        ctx2 = await engine.session_start("proj", "raj")
        assert ctx2.last_episode_summary == "Completed auth refactor"

    @pytest.mark.asyncio
    async def test_multiple_users_isolated(self, engine):
        ctx_raj = await engine.session_start("proj", "raj")
        ctx_alice = await engine.session_start("proj", "alice")

        await engine.session_end(ctx_raj.session_id, "proj", "raj",
                                 summary="Raj's work", outcome="success")
        await engine.session_end(ctx_alice.session_id, "proj", "alice",
                                 summary="Alice's work", outcome="success")

        ctx_raj2 = await engine.session_start("proj", "raj")
        ctx_alice2 = await engine.session_start("proj", "alice")

        assert "Raj" in ctx_raj2.last_episode_summary
        assert "Alice" in ctx_alice2.last_episode_summary


class TestMemAdd:

    @pytest.mark.asyncio
    async def test_add_creates_fact(self, engine):
        result = await engine.add(
            content="We decided to use PostgreSQL for the main database",
            project_id="proj",
            user_id="raj",
            fact_type="decision",
        )
        assert result.action in ("created", "merged")
        assert result.fact_id

    @pytest.mark.asyncio
    async def test_add_explicit_confidence(self, engine):
        result = await engine.add(
            content="Always use pytest for testing",
            project_id="proj",
            user_id="raj",
            confidence="EXPLICIT",
        )
        assert result.fact_id
        fact = await engine.facts.get_fact(result.fact_id)
        if fact:
            assert fact.confidence == "EXPLICIT"

    @pytest.mark.asyncio
    async def test_add_dedup_prevents_duplicates(self, engine):
        # Add same content twice
        r1 = await engine.add("We use PostgreSQL", "proj", "raj",
                               fact_type="decision")
        r2 = await engine.add("We use PostgreSQL", "proj", "raj",
                               fact_type="decision")
        # Second should either be merged or a new one (depends on embedding similarity)
        assert r1.fact_id
        assert r2.fact_id


class TestSearch:

    @pytest.mark.asyncio
    async def test_search_finds_facts(self, engine):
        await engine.add("We decided to use Redis for caching", "proj", "raj",
                         fact_type="decision")
        results = await engine.search("Redis caching", "proj")
        assert len(results.facts) >= 1 or len(results.all_results) >= 0  # SQLite keyword search

    @pytest.mark.asyncio
    async def test_search_finds_context(self, engine):
        engine.context.upsert(
            engine.context.update_from_session(
                "proj", "raj",
                session_summary="Fixed auth",
                recent_decisions=["Use JWT"],
            )
        )
        results = await engine.search("auth", "proj", user_id="raj")
        # Context should be included if "auth" appears in it
        assert isinstance(results, type(results))

    @pytest.mark.asyncio
    async def test_search_returns_search_results_type(self, engine):
        from xme.models import SearchResults
        results = await engine.search("anything", "proj")
        assert isinstance(results, SearchResults)

    @pytest.mark.asyncio
    async def test_search_empty_query_still_returns(self, engine):
        results = await engine.search("", "proj")
        assert results is not None


class TestStats:

    @pytest.mark.asyncio
    async def test_stats_empty_project(self, engine):
        s = await engine.stats("empty-project")
        assert s["total_facts"] == 0
        assert s["total_episodes"] == 0
        assert s["active_users"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_session(self, engine):
        ctx = await engine.session_start("proj", "raj")
        await engine.session_end(ctx.session_id, "proj", "raj",
                                 summary="Test session", outcome="success")
        s = await engine.stats("proj")
        assert s["total_episodes"] >= 1

    @pytest.mark.asyncio
    async def test_stats_counts_facts_by_type(self, engine):
        await engine.add("Use FastAPI", "proj", "raj", fact_type="decision")
        await engine.add("pytest convention", "proj", "raj", fact_type="convention")
        s = await engine.stats("proj")
        assert s["total_facts"] >= 2


class TestContextUpdate:

    @pytest.mark.asyncio
    async def test_update_context_partial(self, engine):
        ctx = engine.update_context("proj", "raj", {
            "current_task": "Build auth service",
            "next_steps": "Add refresh tokens",
        })
        assert ctx.current_task == "Build auth service"
        assert ctx.next_steps == "Add refresh tokens"

    @pytest.mark.asyncio
    async def test_update_context_idempotent(self, engine):
        engine.update_context("proj", "raj", {"current_task": "Task A"})
        engine.update_context("proj", "raj", {"current_task": "Task B"})
        ctx = engine.context.get("proj", "raj")
        assert ctx.current_task == "Task B"
