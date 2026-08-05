"""XME MCP tool definitions and handlers (11 tools)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from mcp.types import Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

XME_TOOLS: list[Tool] = [
    Tool(
        name="xme_session_start",
        description=(
            "Start a memory session for a project + user. "
            "Returns a primed context block with recent decisions, last session summary, "
            "and working context. Inject this into the agent system prompt."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project identifier"},
                "user_id":    {"type": "string", "description": "User identifier"},
                "session_id": {"type": "string", "description": "Optional: reuse a session ID"},
            },
            "required": ["project_id", "user_id"],
        },
    ),
    Tool(
        name="xme_session_end",
        description=(
            "End the current session. Persists the episode, extracts facts, "
            "updates working context. Call at the end of every agent turn / on agentStop."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "user_id":    {"type": "string"},
                "session_id": {"type": "string"},
                "summary":    {"type": "string", "description": "What was accomplished"},
                "outcome": {
                    "type": "string",
                    "enum": ["success", "failed", "partial", "unknown"],
                    "default": "unknown",
                },
                "files_touched": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Files modified this session",
                },
                "next_steps": {"type": "string", "default": ""},
            },
            "required": ["project_id", "user_id", "session_id"],
        },
    ),
    Tool(
        name="xme_add",
        description=(
            "Add content to memory (Mem0-style UPSERT). "
            "Extracts structured facts and deduplicates against existing memory. "
            "Use for explicit decisions, lessons learned, or any notable context."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "user_id":    {"type": "string"},
                "content":    {"type": "string", "description": "Content to remember"},
                "fact_type": {
                    "type": "string",
                    "enum": ["decision", "attempt", "preference", "convention", "entity"],
                    "description": "Force a specific fact type (optional)",
                },
                "metadata":   {"type": "object", "description": "Additional structured data"},
                "session_id": {"type": "string", "description": "Link to session (optional)"},
            },
            "required": ["project_id", "user_id", "content"],
        },
    ),
    Tool(
        name="xme_search",
        description=(
            "Search across all memory layers: episodic (past sessions), "
            "facts (decisions/attempts), and working context. "
            "Returns ranked results from all layers."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "query":      {"type": "string"},
                "user_id":    {"type": "string", "default": ""},
                "layers": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["episodic", "facts", "context"]},
                    "description": "Layers to search (default: all)",
                },
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["project_id", "query"],
        },
    ),
    Tool(
        name="xme_get_context",
        description=(
            "Get the current working context for a project+user. "
            "Returns a markdown block ready for prompt injection."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "user_id":    {"type": "string"},
            },
            "required": ["project_id", "user_id"],
        },
    ),
    Tool(
        name="xme_facts",
        description=(
            "Query the fact graph. List decisions, attempts, preferences, "
            "conventions, or entities for a project."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "fact_type": {
                    "type": "string",
                    "enum": ["decision", "attempt", "preference", "convention", "entity", ""],
                    "default": "",
                },
                "user_id":  {"type": "string", "default": ""},
                "query":    {"type": "string", "default": "", "description": "Optional keyword filter"},
                "limit":    {"type": "integer", "default": 20},
            },
            "required": ["project_id"],
        },
    ),
    Tool(
        name="xme_episodes",
        description=(
            "Search past sessions (episodic memory). "
            "Full-text + semantic search across all session transcripts."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "query":      {"type": "string", "description": "Search query"},
                "user_id":    {"type": "string", "default": ""},
                "date_from":  {"type": "string", "description": "ISO date filter start"},
                "date_to":    {"type": "string", "description": "ISO date filter end"},
                "limit":      {"type": "integer", "default": 10},
            },
            "required": ["project_id", "query"],
        },
    ),
    Tool(
        name="xme_remember",
        description=(
            "Explicitly remember a specific fact with type and metadata. "
            "More precise than xme_add — use when you know exactly what type of fact it is."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "user_id":    {"type": "string"},
                "fact_type": {
                    "type": "string",
                    "enum": ["decision", "attempt", "preference", "convention", "entity"],
                },
                "title":   {"type": "string"},
                "content": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "description": (
                        "Type-specific fields. "
                        "decision: {outcome, alternatives_considered}. "
                        "attempt: {result, failure_reason, lessons_learned}. "
                        "preference: {key, value}."
                    ),
                },
            },
            "required": ["project_id", "user_id", "fact_type", "title", "content"],
        },
    ),
    Tool(
        name="xme_forget",
        description="Soft-delete a memory node (marks as deleted, not physically removed).",
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "fact_id":    {"type": "string", "description": "ID of the fact to delete"},
            },
            "required": ["project_id", "fact_id"],
        },
    ),
    Tool(
        name="xme_export",
        description=(
            "Export project memory to Obsidian vault, wiki markdown, or "
            "Graphify-compatible graph.json."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "format": {
                    "type": "string",
                    "enum": ["obsidian", "wiki", "graphify"],
                    "default": "obsidian",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Output directory (default: .xanther/{format})",
                    "default": "",
                },
            },
            "required": ["project_id"],
        },
    ),
    Tool(
        name="xme_context_update",
        description=(
            "Update the working context for a project+user. "
            "UPSERT semantics — only specified fields are changed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "user_id":    {"type": "string"},
                "updates": {
                    "type": "object",
                    "description": (
                        "Fields to update: current_task, open_questions, "
                        "next_steps, blockers, files_in_focus"
                    ),
                },
            },
            "required": ["project_id", "user_id", "updates"],
        },
    ),
]

_TOOL_NAMES = {t.name for t in XME_TOOLS}


def is_xme_tool(name: str) -> bool:
    return name in _TOOL_NAMES


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class XMEToolHandler:
    """Routes MCP tool calls to the MemoryEngine."""

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        from xme.engine import get_engine
        engine = await get_engine()

        try:
            if name == "xme_session_start":
                return await self._session_start(engine, args)
            elif name == "xme_session_end":
                return await self._session_end(engine, args)
            elif name == "xme_add":
                return await self._add(engine, args)
            elif name == "xme_search":
                return await self._search(engine, args)
            elif name == "xme_get_context":
                return await self._get_context(engine, args)
            elif name == "xme_facts":
                return await self._facts(engine, args)
            elif name == "xme_episodes":
                return await self._episodes(engine, args)
            elif name == "xme_remember":
                return await self._remember(engine, args)
            elif name == "xme_forget":
                return await self._forget(engine, args)
            elif name == "xme_export":
                return await self._export(engine, args)
            elif name == "xme_context_update":
                return await self._context_update(engine, args)
            else:
                return {"error": f"Unknown XME tool: {name}"}
        except Exception as e:
            logger.exception("XME tool %s failed", name)
            return {"error": str(e), "tool": name}

    async def _session_start(self, engine, args):
        ctx = await engine.session_start(
            project_id=args["project_id"],
            user_id=args["user_id"],
            session_id=args.get("session_id"),
        )
        return {
            "session_id": ctx.session_id,
            "prompt_block": ctx.prompt_block,
            "recent_facts": [f.to_dict() for f in ctx.recent_facts],
            "last_episode_summary": ctx.last_episode_summary,
        }

    async def _session_end(self, engine, args):
        ep = await engine.session_end(
            session_id=args["session_id"],
            project_id=args["project_id"],
            user_id=args["user_id"],
            summary=args.get("summary", ""),
            outcome=args.get("outcome", "unknown"),
            files_touched=args.get("files_touched"),
            next_steps=args.get("next_steps", ""),
        )
        return {
            "status": "ok",
            "episode_id": ep.episode_id,
            "facts_extracted": len(ep.fact_ids),
            "summary": ep.summary,
        }

    async def _add(self, engine, args):
        result = await engine.add(
            content=args["content"],
            project_id=args["project_id"],
            user_id=args["user_id"],
            fact_type=args.get("fact_type"),
            metadata=args.get("metadata"),
            session_id=args.get("session_id"),
        )
        return {"status": "ok", "action": result.action, "fact_id": result.fact_id}

    async def _search(self, engine, args):
        results = await engine.search(
            query=args["query"],
            project_id=args["project_id"],
            user_id=args.get("user_id") or None,
            layers=args.get("layers"),
            limit=int(args.get("limit", 10)),
        )
        return {
            "query": results.query,
            "total": len(results.all_results),
            "episodic": [_fmt_result(r) for r in results.episodic],
            "facts": [_fmt_result(r) for r in results.facts],
            "context": [_fmt_result(r) for r in results.context],
        }

    async def _get_context(self, engine, args):
        block = engine.get_context(args["project_id"], args["user_id"])
        ctx = engine.context.get(args["project_id"], args["user_id"])
        return {
            "prompt_block": block,
            "context": ctx.to_dict() if ctx else None,
        }

    async def _facts(self, engine, args):
        fact_type = args.get("fact_type") or None
        query = args.get("query", "")
        if query:
            results = await engine.facts.search_facts(
                query=query,
                project_id=args["project_id"],
                fact_type=fact_type,
                limit=int(args.get("limit", 20)),
            )
            facts_data = [r.data for r in results]
        else:
            facts = await engine.facts.list_facts(
                project_id=args["project_id"],
                fact_type=fact_type,
                user_id=args.get("user_id") or None,
                limit=int(args.get("limit", 20)),
            )
            facts_data = [f.to_dict() for f in facts]
        return {"project_id": args["project_id"], "facts": facts_data, "count": len(facts_data)}

    async def _episodes(self, engine, args):
        results = await engine.episodic.search(
            query=args["query"],
            project_id=args["project_id"],
            user_id=args.get("user_id") or None,
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            limit=int(args.get("limit", 10)),
        )
        return {
            "query": args["query"],
            "episodes": [_fmt_result(r) for r in results],
            "count": len(results),
        }

    async def _remember(self, engine, args):
        from xme.models import Confidence, Fact
        fact = Fact(
            fact_type=args["fact_type"],
            project_id=args["project_id"],
            user_id=args["user_id"],
            title=args["title"],
            content=args["content"],
            metadata=args.get("metadata", {}),
            confidence=Confidence.EXPLICIT.value,
        )
        result = await engine.facts.upsert_fact(fact, engine._embedder)
        return {"status": "ok", "action": result.action, "fact_id": result.fact_id}

    async def _forget(self, engine, args):
        await engine.facts.delete_fact(args["fact_id"])
        return {"status": "ok", "fact_id": args["fact_id"]}

    async def _export(self, engine, args):
        from xme.export import run_export
        output_path = await run_export(
            engine=engine,
            project_id=args["project_id"],
            fmt=args.get("format", "obsidian"),
            output_dir=args.get("output_dir") or None,
        )
        return {"status": "ok", "output_path": str(output_path)}

    async def _context_update(self, engine, args):
        ctx = engine.update_context(
            project_id=args["project_id"],
            user_id=args["user_id"],
            updates=args["updates"],
        )
        return {"status": "ok", "context": ctx.to_dict()}


def _fmt_result(r) -> dict:  # type: ignore[type-arg]
    return {
        "layer": r.layer,
        "id": r.item_id,
        "score": round(r.score, 3),
        "summary": r.summary,
        "highlight": r.highlight,
    }
