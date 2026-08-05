"""Integration tests for XME MCP tool handlers."""
import pytest
from xme.config import XMESettings
from xme.engine import MemoryEngine, reset_engine
from xme.server.mcp_tools import XMEToolHandler


@pytest.fixture(autouse=True)
def reset():
    reset_engine()
    yield
    reset_engine()


@pytest.fixture
async def handler_and_engine(tmp_path):
    settings = XMESettings(
        sqlite_path=str(tmp_path / "xme.db"),
        fallback_mode=True,
        opensearch_enabled=False,
    )
    engine = MemoryEngine(settings)
    await engine.initialize()

    # Patch get_engine to return our test engine
    import xme.engine as eng_mod
    orig = eng_mod._engine
    eng_mod._engine = engine

    handler = XMEToolHandler()
    yield handler, engine

    eng_mod._engine = orig
    await engine.shutdown()


class TestMCPSessionTools:

    @pytest.mark.asyncio
    async def test_session_start(self, handler_and_engine):
        handler, _ = handler_and_engine
        result = await handler.dispatch("xme_session_start", {
            "project_id": "test-proj", "user_id": "raj"
        })
        assert "session_id" in result
        assert "prompt_block" in result
        assert "recent_facts" in result

    @pytest.mark.asyncio
    async def test_session_end(self, handler_and_engine):
        handler, _ = handler_and_engine
        # Start first
        start = await handler.dispatch("xme_session_start", {
            "project_id": "test-proj", "user_id": "raj"
        })
        # End
        end = await handler.dispatch("xme_session_end", {
            "project_id": "test-proj",
            "user_id": "raj",
            "session_id": start["session_id"],
            "summary": "Fixed auth bug",
            "outcome": "success",
        })
        assert end["status"] == "ok"
        assert "episode_id" in end
        assert "facts_extracted" in end


class TestMCPMemoryTools:

    @pytest.mark.asyncio
    async def test_xme_add(self, handler_and_engine):
        handler, _ = handler_and_engine
        result = await handler.dispatch("xme_add", {
            "project_id": "proj", "user_id": "raj",
            "content": "We decided to use FastAPI for the backend API",
            "fact_type": "decision",
        })
        assert result["status"] == "ok"
        assert result["action"] in ("created", "merged")
        assert "fact_id" in result

    @pytest.mark.asyncio
    async def test_xme_remember(self, handler_and_engine):
        handler, _ = handler_and_engine
        result = await handler.dispatch("xme_remember", {
            "project_id": "proj", "user_id": "raj",
            "fact_type": "decision",
            "title": "Use PostgreSQL",
            "content": "PostgreSQL chosen for ACID compliance",
            "metadata": {"outcome": "validated"},
        })
        assert result["status"] == "ok"
        assert result["action"] in ("created", "merged")

    @pytest.mark.asyncio
    async def test_xme_facts_list(self, handler_and_engine):
        handler, _ = handler_and_engine
        # Add some facts first
        await handler.dispatch("xme_remember", {
            "project_id": "proj", "user_id": "raj",
            "fact_type": "decision", "title": "Use FastAPI", "content": "For async support",
        })
        result = await handler.dispatch("xme_facts", {"project_id": "proj"})
        assert result["count"] >= 1
        assert isinstance(result["facts"], list)

    @pytest.mark.asyncio
    async def test_xme_search(self, handler_and_engine):
        handler, _ = handler_and_engine
        await handler.dispatch("xme_add", {
            "project_id": "proj", "user_id": "raj",
            "content": "Authentication uses JWT refresh tokens",
        })
        result = await handler.dispatch("xme_search", {
            "project_id": "proj", "query": "authentication JWT",
        })
        assert "total" in result
        assert "facts" in result
        assert "episodic" in result

    @pytest.mark.asyncio
    async def test_xme_get_context(self, handler_and_engine):
        handler, engine = handler_and_engine
        engine.context.update_from_session(
            "proj", "raj",
            session_summary="Fixed auth",
            recent_decisions=["Use JWT"],
        )
        result = await handler.dispatch("xme_get_context", {
            "project_id": "proj", "user_id": "raj"
        })
        assert "prompt_block" in result
        assert "context" in result
        assert result["context"] is not None

    @pytest.mark.asyncio
    async def test_xme_forget(self, handler_and_engine):
        handler, _ = handler_and_engine
        add_result = await handler.dispatch("xme_remember", {
            "project_id": "proj", "user_id": "raj",
            "fact_type": "entity", "title": "Redis", "content": "Cache layer",
        })
        fact_id = add_result["fact_id"]
        forget_result = await handler.dispatch("xme_forget", {
            "project_id": "proj", "fact_id": fact_id,
        })
        assert forget_result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_xme_context_update(self, handler_and_engine):
        handler, _ = handler_and_engine
        result = await handler.dispatch("xme_context_update", {
            "project_id": "proj", "user_id": "raj",
            "updates": {"current_task": "Build auth service", "next_steps": "Deploy"},
        })
        assert result["status"] == "ok"
        assert result["context"]["current_task"] == "Build auth service"

    @pytest.mark.asyncio
    async def test_xme_episodes_returns_list(self, handler_and_engine):
        handler, _ = handler_and_engine
        result = await handler.dispatch("xme_episodes", {
            "project_id": "proj", "query": "auth",
        })
        assert "episodes" in result
        assert isinstance(result["episodes"], list)

    @pytest.mark.asyncio
    async def test_xme_export_obsidian(self, handler_and_engine, tmp_path):
        handler, _ = handler_and_engine
        await handler.dispatch("xme_remember", {
            "project_id": "proj", "user_id": "raj",
            "fact_type": "decision", "title": "Use FastAPI", "content": "For async support",
        })
        result = await handler.dispatch("xme_export", {
            "project_id": "proj",
            "format": "obsidian",
            "output_dir": str(tmp_path / "obsidian"),
        })
        assert result["status"] == "ok"
        assert "output_path" in result

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, handler_and_engine):
        handler, _ = handler_and_engine
        result = await handler.dispatch("xme_nonexistent", {"project_id": "proj"})
        assert "error" in result
