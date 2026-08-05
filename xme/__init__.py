"""XME — Xanther Memory Engine.

Generic, agent-agnostic persistent memory for AI coding assistants.
Three layers: Episodic (OpenSearch) + Facts (Neo4j) + Context (SQLite).

Quick start::

    from xme import get_engine

    engine = await get_engine()

    # Start session → get primed context for prompt injection
    ctx = await engine.session_start("my-project", "raj")
    print(ctx.prompt_block)

    # Record turns (non-blocking)
    engine.record_turn(ctx.session_id, "user", "How does auth work?")
    engine.record_turn(ctx.session_id, "assistant", "Auth uses JWT...")

    # Add explicit memory (Mem0-style UPSERT)
    await engine.add("We decided to use FastAPI", "my-project", "raj",
                     fact_type="decision")

    # Search all layers
    results = await engine.search("auth decisions", "my-project")

    # End session → persist + extract facts
    await engine.session_end(ctx.session_id, "my-project", "raj",
                             summary="Refactored auth", outcome="success")
"""

from xme.models import (
    Episode,
    Fact,
    FactType,
    Confidence,
    Outcome,
    Turn,
    WorkingContext,
    SessionContext,
    SearchResults,
    MemorySearchResult,
    UpsertResult,
)
from xme.config import XMESettings, get_settings
from xme.engine import MemoryEngine, get_engine, reset_engine

__all__ = [
    "Episode",
    "Fact",
    "FactType",
    "Confidence",
    "Outcome",
    "Turn",
    "WorkingContext",
    "SessionContext",
    "SearchResults",
    "MemorySearchResult",
    "UpsertResult",
    "XMESettings",
    "get_settings",
    "MemoryEngine",
    "get_engine",
    "reset_engine",
]

__version__ = "0.1.0"
