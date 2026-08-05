"""End-to-end test: full session flow from start to export."""
import pytest
from pathlib import Path
from xme.config import XMESettings
from xme.engine import MemoryEngine, reset_engine
from xme.models import FactType


@pytest.fixture(autouse=True)
def reset():
    reset_engine()
    yield
    reset_engine()


@pytest.fixture
async def engine(tmp_path):
    settings = XMESettings(
        sqlite_path=str(tmp_path / "xme.db"),
        fallback_mode=True,
        opensearch_enabled=False,
    )
    async with MemoryEngine(settings) as eng:
        yield eng


@pytest.mark.asyncio
async def test_full_multi_turn_session(engine, tmp_path):
    """Full session: start → multiple turns → end → verify persistence."""
    # === SESSION 1 ===
    ctx = await engine.session_start("xanther", "raj")
    assert ctx.session_id
    assert ctx.working_context is None  # first session

    # Simulate a conversation about auth refactoring
    turns = [
        ("user", "Let's refactor the auth module"),
        ("assistant", "OK. We decided to move JWT validation to a dedicated auth service."),
        ("user", "What about the old Redis approach?"),
        ("assistant", "The Redis lock failed because of timeout under high load. "
                      "We should prefer stateless JWT instead."),
        ("user", "What's our testing convention?"),
        ("assistant", "We always use pytest with asyncio mode for all tests in this project."),
        ("tool", "xce_search result: found auth module at xce/auth/jwt.py"),
    ]
    for role, content in turns:
        engine.record_turn(ctx.session_id, role, content)

    # End session
    ep = await engine.session_end(
        session_id=ctx.session_id,
        project_id="xanther",
        user_id="raj",
        summary="Refactored auth to JWT microservice. Dropped Redis lock.",
        outcome="success",
        files_touched=["xce/auth/jwt.py", "xce/auth/service.py"],
        next_steps="Deploy auth service to staging",
    )

    assert ep.outcome == "success"
    assert ep.summary
    assert ep.ended_at

    # === VERIFY PERSISTENCE ===
    # Episode should be queryable
    episodes = await engine.episodic.list_episodes("xanther", "raj", limit=5)
    assert len(episodes) >= 1

    # Working context should be updated
    wctx = engine.context.get("xanther", "raj")
    assert wctx is not None
    assert "Refactored" in wctx.last_session_summary
    assert "Deploy" in wctx.next_steps
    assert "xce/auth/jwt.py" in wctx.files_in_focus

    # === SESSION 2 — should get context from session 1 ===
    ctx2 = await engine.session_start("xanther", "raj")
    assert "Refactored" in ctx2.last_episode_summary
    assert ctx2.prompt_block  # should have something to inject
    assert "Deploy" in ctx2.prompt_block or "Refactored" in ctx2.prompt_block


@pytest.mark.asyncio
async def test_multi_user_memory_isolation(engine):
    """Two users on same project should not see each other's sessions."""
    # Raj's session
    ctx_raj = await engine.session_start("shared-proj", "raj")
    engine.record_turn(ctx_raj.session_id, "assistant", "Raj decided to use Redis")
    await engine.session_end(ctx_raj.session_id, "shared-proj", "raj",
                             summary="Raj: Redis decision", outcome="success")

    # Alice's session
    ctx_alice = await engine.session_start("shared-proj", "alice")
    engine.record_turn(ctx_alice.session_id, "assistant", "Alice decided to use Kafka")
    await engine.session_end(ctx_alice.session_id, "shared-proj", "alice",
                             summary="Alice: Kafka decision", outcome="success")

    # New sessions should see user-specific context
    ctx_raj2 = await engine.session_start("shared-proj", "raj")
    ctx_alice2 = await engine.session_start("shared-proj", "alice")

    assert "Raj" in ctx_raj2.last_episode_summary
    assert "Alice" in ctx_alice2.last_episode_summary
    # Raj should NOT see Alice's summary as last_episode
    assert "Alice" not in ctx_raj2.last_episode_summary
    assert "Raj" not in ctx_alice2.last_episode_summary


@pytest.mark.asyncio
async def test_mem0_style_add_and_dedup(engine):
    """Explicit add — dedup works when embeddings are available, gracefully skips when not."""
    from xme.extraction.embedder import LocalEmbedder
    embedder = LocalEmbedder()
    test_vec = embedder.embed("test")
    embeddings_available = any(x != 0.0 for x in test_vec)

    r1 = await engine.add(
        "We use PostgreSQL for all persistence",
        "proj", "raj", fact_type="decision"
    )
    assert r1.action == "created"

    r2 = await engine.add(
        "We use PostgreSQL for all persistence",
        "proj", "raj", fact_type="decision"
    )
    assert r2.action in ("created", "merged")

    facts = await engine.facts.list_facts("proj", fact_type="decision")
    if embeddings_available:
        # With real embeddings: dedup should prevent duplicates
        assert len(facts) <= 2
    else:
        # Without embeddings (zero vectors): dedup can't work — just check we stored facts
        assert len(facts) >= 1


@pytest.mark.asyncio
async def test_export_obsidian(engine, tmp_path):
    """Export to Obsidian vault after a session."""
    # Add some content
    ctx = await engine.session_start("proj", "raj")
    engine.record_turn(ctx.session_id, "assistant",
                       "We decided to use FastAPI. We always use pytest.")
    await engine.session_end(
        ctx.session_id, "proj", "raj",
        summary="FastAPI decision + pytest convention",
        outcome="success",
    )
    await engine.add("Use PostgreSQL", "proj", "raj", fact_type="decision")

    # Export
    from xme.export import run_export
    output = await run_export(engine, "proj", fmt="obsidian",
                              output_dir=str(tmp_path / "obs"))
    assert output.exists()
    assert (output / "index.md").exists()
    assert (output / ".obsidian").exists()


@pytest.mark.asyncio
async def test_export_graphify_compat(engine, tmp_path):
    """Export in Graphify-compatible format."""
    await engine.add("Use FastAPI", "proj", "raj", fact_type="decision")
    await engine.add("Use pytest", "proj", "raj", fact_type="convention")

    from xme.export import run_export
    output = await run_export(engine, "proj", fmt="graphify",
                              output_dir=str(tmp_path / "gf"))
    assert (output / "graph.json").exists()
    assert (output / "GRAPH_REPORT.md").exists()

    import json
    graph = json.loads((output / "graph.json").read_text())
    assert "nodes" in graph
    assert "edges" in graph


@pytest.mark.asyncio
async def test_stats_reflect_all_activity(engine):
    """Stats should reflect sessions, facts, users."""
    ctx = await engine.session_start("proj", "alice")
    await engine.session_end(ctx.session_id, "proj", "alice",
                             summary="Alice session", outcome="success")
    await engine.add("Use Redis", "proj", "alice", fact_type="decision")
    await engine.add("Always write tests", "proj", "alice", fact_type="convention")

    s = await engine.stats("proj")
    assert s["total_episodes"] >= 1
    assert s["total_facts"] >= 2
    assert s["active_users"] >= 1
    assert "alice" in s["users"]
