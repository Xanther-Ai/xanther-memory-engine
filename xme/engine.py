"""XME MemoryEngine — the single facade for all memory operations.

Usage::

    from xme import get_engine
    engine = await get_engine()

    ctx = await engine.session_start("my-project", "raj")
    # inject ctx.prompt_block into agent system prompt

    engine.record_turn(ctx.session_id, "user", "How does auth work?")
    engine.record_turn(ctx.session_id, "assistant", "Auth uses JWT...")

    await engine.session_end(
        session_id=ctx.session_id,
        project_id="my-project",
        user_id="raj",
        summary="Discussed JWT auth",
        outcome="success",
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from xme.config import XMESettings, get_settings
from xme.extraction.embedder import LocalEmbedder
from xme.extraction.extractor import FactExtractor
from xme.layers.context import ContextStore
from xme.layers.episodic import EpisodicStore
from xme.layers.facts import FactGraphStore
from xme.models import (
    Episode,
    Fact,
    FactType,
    MemorySearchResult,
    SearchResults,
    SessionContext,
    Turn,
    UpsertResult,
    WorkingContext,
    _now,
    _uid,
)

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Unified entry point for XME — manages all 3 layers."""

    def __init__(self, settings: XMESettings) -> None:
        self._settings = settings
        self._embedder = LocalEmbedder(settings.embedding_model)
        self._extractor = FactExtractor(
            llm_api_key=settings.llm_api_key,
            llm_model=settings.llm_model,
            llm_base_url=settings.llm_base_url,
        )

        db_path = settings.resolved_sqlite_path()

        self.episodic = EpisodicStore(
            opensearch_url=settings.opensearch_url,
            sqlite_path=str(db_path),
            opensearch_enabled=settings.opensearch_enabled and not settings.fallback_mode,
            embedding_dims=settings.embedding_dims,
        )
        self.facts = FactGraphStore(
            neo4j_uri=settings.neo4j_uri,
            neo4j_auth=settings.neo4j_auth,
            sqlite_path=str(db_path),
            dedup_threshold=settings.dedup_threshold,
            embedding_dims=settings.embedding_dims,
            neo4j_enabled=not settings.fallback_mode,
        )
        self.context = ContextStore(db_path)

        # Active sessions: session_id → Episode (being built)
        self._active: dict[str, Episode] = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._initialized:
            return
        self.episodic.connect()
        await self.facts.connect()
        self.context.connect()
        self._initialized = True
        logger.info("XME MemoryEngine initialized (fallback=%s)", self._settings.fallback_mode)

    async def shutdown(self) -> None:
        self.episodic.close()
        await self.facts.close()
        self.context.close()
        self._initialized = False

    async def __aenter__(self) -> "MemoryEngine":
        await self.initialize()
        return self

    async def __aexit__(self, *_) -> None:  # type: ignore[override]
        await self.shutdown()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def session_start(
        self,
        project_id: str,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> SessionContext:
        """Start a session. Returns primed context for agent prompt injection."""
        sid = session_id or _uid()

        # Create active episode
        ep = Episode(
            episode_id=_uid(),
            project_id=project_id,
            user_id=user_id,
            session_id=sid,
            started_at=_now(),
        )
        self._active[sid] = ep

        # Load working context (Layer 3)
        wctx = self.context.get(project_id, user_id)

        # Load recent facts (Layer 2)
        recent_facts = await self.facts.list_facts(project_id, limit=5)

        # Load last episode summary (Layer 1)
        last_eps = await self.episodic.list_episodes(project_id, user_id, limit=1)
        last_summary = last_eps[0].summary if last_eps else ""

        # Build prompt block
        prompt_parts = []
        if wctx:
            prompt_parts.append(wctx.as_prompt_block())
        elif last_summary:
            prompt_parts.append(f"**Last session**: {last_summary}")

        if recent_facts:
            prompt_parts.append("\n**Recent facts**:")
            for f in recent_facts[:5]:
                prompt_parts.append(f"- [{f.fact_type}] {f.title}")

        prompt_block = "\n".join(prompt_parts)

        return SessionContext(
            session_id=sid,
            project_id=project_id,
            user_id=user_id,
            working_context=wctx,
            recent_facts=recent_facts,
            last_episode_summary=last_summary,
            prompt_block=prompt_block,
        )

    def record_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
        tool_result: Optional[str] = None,
    ) -> None:
        """Record a single turn. Non-blocking — buffered in memory."""
        turn = Turn(
            role=role,
            content=content[:2000],  # cap length to avoid bloat
            timestamp=_now(),
            tool_name=tool_name,
            tool_result=tool_result[:500] if tool_result else None,
        )
        ep = self._active.get(session_id)
        if ep:
            ep.turns.append(turn)
        self.episodic.append_turn(session_id, turn)

    async def session_end(
        self,
        session_id: str,
        project_id: str,
        user_id: str,
        summary: str = "",
        outcome: str = "unknown",
        files_touched: Optional[list[str]] = None,
        next_steps: str = "",
    ) -> Episode:
        """End session: persist episode, extract facts, update context."""
        ep = self._active.pop(session_id, None)
        if ep is None:
            # Reconstruct minimal episode
            ep = Episode(
                project_id=project_id,
                user_id=user_id,
                session_id=session_id,
            )

        ep.ended_at = _now()
        ep.summary = summary or _auto_summary(ep)
        ep.outcome = outcome

        # 1. Persist episode (Layer 1)
        ep.embedding = self._embedder.embed(ep.summary) if ep.summary else []
        await self.episodic.save_episode(ep)

        # 2. Extract facts from transcript (Layer 2)
        facts: list[Fact] = []
        if ep.full_transcript:
            facts = await self._extractor.extract(
                ep.full_transcript, project_id, user_id, ep.episode_id
            )
            upsert_results: list[UpsertResult] = []
            for fact in facts:
                result = await self.facts.upsert_fact(fact, self._embedder)
                upsert_results.append(result)
                if result.action == "created":
                    ep.fact_ids.append(result.fact_id)

            # Link episode → facts in Neo4j
            if ep.fact_ids:
                await self.facts.link_episode_to_facts(ep.episode_id, ep.fact_ids)

        # 3. Update working context (Layer 3)
        decision_titles = [
            f.title for f in facts
            if f.fact_type == FactType.DECISION.value
        ]
        self.context.update_from_session(
            project_id=project_id,
            user_id=user_id,
            session_summary=ep.summary,
            recent_decisions=decision_titles,
            next_steps=next_steps,
            files_touched=files_touched,
        )

        logger.info(
            "Session ended: project=%s user=%s facts_extracted=%d outcome=%s",
            project_id, user_id, len(ep.fact_ids), outcome,
        )
        return ep

    # ------------------------------------------------------------------
    # Mem0-style add (explicit UPSERT)
    # ------------------------------------------------------------------

    async def add(
        self,
        content: str,
        project_id: str,
        user_id: str,
        fact_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        confidence: str = "EXPLICIT",
        session_id: Optional[str] = None,
    ) -> UpsertResult:
        """Add content to memory — extract facts and upsert."""
        facts = await self._extractor.extract(content, project_id, user_id, session_id)

        if not facts:
            # No structured extraction — store as generic entity
            facts = [Fact(
                fact_type=fact_type or FactType.ENTITY.value,
                project_id=project_id,
                user_id=user_id,
                title=content[:100],
                content=content,
                metadata=metadata or {},
                source_episode_id=session_id,
                confidence=confidence,
            )]

        results: list[UpsertResult] = []
        for fact in facts:
            if fact_type:
                fact.fact_type = fact_type
            if metadata:
                fact.metadata.update(metadata)
            fact.confidence = confidence
            result = await self.facts.upsert_fact(fact, self._embedder)
            results.append(result)

        # Return the first/primary result
        return results[0] if results else UpsertResult(action="created", fact_id=_uid())

    # ------------------------------------------------------------------
    # Search across all layers
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        project_id: str,
        user_id: Optional[str] = None,
        layers: Optional[list[str]] = None,
        limit: int = 20,
    ) -> SearchResults:
        """Unified search across episodic + facts + context."""
        layers = layers or ["episodic", "facts", "context"]
        results = SearchResults(query=query, project_id=project_id)

        # Embed query once for semantic search
        query_emb = self._embedder.embed(query)

        if "episodic" in layers:
            results.episodic = await self.episodic.search(
                query=query,
                project_id=project_id,
                user_id=user_id,
                limit=limit,
                query_embedding=query_emb,
            )

        if "facts" in layers:
            results.facts = await self.facts.search_facts(
                query=query,
                project_id=project_id,
                limit=limit,
                query_embedding=query_emb,
            )

        if "context" in layers:
            ctx = self.context.get(project_id, user_id or "")
            if ctx:
                text = ctx.as_prompt_block()
                if query.lower() in text.lower():
                    results.context = [MemorySearchResult(
                        layer="context",
                        item_id=f"{project_id}:{user_id}",
                        score=0.9,
                        summary=ctx.current_task or ctx.last_session_summary,
                        data=ctx.to_dict(),
                    )]

        return results

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def get_context(self, project_id: str, user_id: str) -> str:
        """Get formatted context block for prompt injection."""
        return self.context.get_prompt_block(project_id, user_id)

    def update_context(
        self,
        project_id: str,
        user_id: str,
        updates: dict[str, Any],
    ) -> WorkingContext:
        """Partial UPSERT of working context fields."""
        ctx = self.context.get(project_id, user_id) or WorkingContext(
            project_id=project_id, user_id=user_id
        )
        for k, v in updates.items():
            if hasattr(ctx, k):
                setattr(ctx, k, v)
        ctx.updated_at = _now()
        self.context.upsert(ctx)
        return ctx

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def stats(self, project_id: str) -> dict[str, Any]:
        facts = await self.facts.list_facts(project_id, limit=1000)
        episodes = await self.episodic.list_episodes(project_id, limit=1000)
        users = self.context.list_users(project_id)
        fact_types: dict[str, int] = {}
        for f in facts:
            fact_types[f.fact_type] = fact_types.get(f.fact_type, 0) + 1
        return {
            "project_id": project_id,
            "total_facts": len(facts),
            "fact_types": fact_types,
            "total_episodes": len(episodes),
            "active_users": len(users),
            "users": users,
            "last_activity": episodes[0].ended_at if episodes else None,
        }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_engine: Optional[MemoryEngine] = None


async def get_engine(settings: Optional[XMESettings] = None) -> MemoryEngine:
    """Get or create the global MemoryEngine singleton."""
    global _engine
    if _engine is None:
        _engine = MemoryEngine(settings or get_settings())
        await _engine.initialize()
    return _engine


def reset_engine() -> None:
    """Force re-create engine (useful in tests)."""
    global _engine
    _engine = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_summary(ep: Episode) -> str:
    """Generate a basic summary from the episode transcript."""
    if not ep.turns:
        return ""
    # Take first user message + first assistant response
    user_turns = [t for t in ep.turns if t.role == "user"]
    assistant_turns = [t for t in ep.turns if t.role == "assistant"]
    parts = []
    if user_turns:
        parts.append(f"Task: {user_turns[0].content[:100]}")
    if assistant_turns:
        parts.append(f"Done: {assistant_turns[-1].content[:100]}")
    return " | ".join(parts) or f"Session with {len(ep.turns)} turns"
