"""XME domain models — Episodes, Facts, WorkingContext.

Three memory layers:
  Episode      → Layer 1 (episodic): verbatim session transcript
  Fact         → Layer 2 (facts): structured knowledge graph node
  WorkingContext → Layer 3 (context): per-(project,user) UPSERT state
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

class FactType(str, Enum):
    DECISION   = "decision"
    ATTEMPT    = "attempt"
    PREFERENCE = "preference"
    CONVENTION = "convention"
    ENTITY     = "entity"

class Confidence(str, Enum):
    EXPLICIT  = "EXPLICIT"   # user explicitly stated
    EXTRACTED = "EXTRACTED"  # LLM or regex extracted from text
    INFERRED  = "INFERRED"   # agent inferred from context


class Outcome(str, Enum):
    SUCCESS  = "success"
    FAILED   = "failed"
    PARTIAL  = "partial"
    UNKNOWN  = "unknown"


# ---------------------------------------------------------------------------
# Layer 1 — Episodic
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single exchange within a session."""
    role: str                    # "user" | "assistant" | "tool" | "note"
    content: str
    timestamp: str = field(default_factory=_now)
    tool_name: Optional[str] = None
    tool_result: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "tool_result": self.tool_result,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        return cls(
            role=d["role"],
            content=d["content"],
            timestamp=d.get("timestamp", _now()),
            tool_name=d.get("tool_name"),
            tool_result=d.get("tool_result"),
        )

    def as_text(self) -> str:
        if self.role == "tool":
            return f"[Tool: {self.tool_name}] {self.content}"
        prefix = "User" if self.role == "user" else "Assistant"
        return f"{prefix}: {self.content}"


@dataclass
class Episode:
    """A complete agent session — verbatim transcript + metadata."""
    episode_id: str = field(default_factory=_uid)
    project_id: str = ""
    user_id: str = ""
    session_id: str = field(default_factory=_uid)
    started_at: str = field(default_factory=_now)
    ended_at: Optional[str] = None
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    outcome: str = Outcome.UNKNOWN.value
    tags: list[str] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)  # linked Fact IDs
    embedding: list[float] = field(default_factory=list)

    @property
    def full_transcript(self) -> str:
        return "\n".join(t.as_text() for t in self.turns)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "turns": [t.to_dict() for t in self.turns],
            "full_transcript": self.full_transcript,
            "summary": self.summary,
            "outcome": self.outcome,
            "tags": self.tags,
            "fact_ids": self.fact_ids,
            "turn_count": self.turn_count,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Episode":
        ep = cls(
            episode_id=d.get("episode_id", _uid()),
            project_id=d.get("project_id", ""),
            user_id=d.get("user_id", ""),
            session_id=d.get("session_id", _uid()),
            started_at=d.get("started_at", _now()),
            ended_at=d.get("ended_at"),
            summary=d.get("summary", ""),
            outcome=d.get("outcome", Outcome.UNKNOWN.value),
            tags=d.get("tags", []),
            fact_ids=d.get("fact_ids", []),
            embedding=d.get("embedding", []),
        )
        ep.turns = [Turn.from_dict(t) for t in d.get("turns", [])]
        return ep


# ---------------------------------------------------------------------------
# Layer 2 — Facts
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    """A structured knowledge node extracted from memory."""
    fact_id: str = field(default_factory=_uid)
    fact_type: str = FactType.ENTITY.value
    project_id: str = ""
    user_id: str = ""
    title: str = ""
    content: str = ""                             # free-text description
    metadata: dict[str, Any] = field(default_factory=dict)  # type-specific fields
    source_episode_id: Optional[str] = None
    related_fact_ids: list[str] = field(default_factory=list)
    code_node_ids: list[str] = field(default_factory=list)  # XCE ASTNode IDs
    confidence: str = Confidence.EXTRACTED.value
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    embedding: list[float] = field(default_factory=list)
    status: str = "active"    # "active" | "deleted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "fact_type": self.fact_type,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "source_episode_id": self.source_episode_id,
            "related_fact_ids": self.related_fact_ids,
            "code_node_ids": self.code_node_ids,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fact":
        f = cls()
        for k, v in d.items():
            if k == "embedding":
                continue  # don't restore embedding from dict
            if hasattr(f, k):
                setattr(f, k, v)
        return f

    def text_for_embedding(self) -> str:
        return f"{self.fact_type}: {self.title}. {self.content}"


@dataclass
class UpsertResult:
    """Result of a fact upsert operation."""
    action: str       # "created" | "merged"
    fact_id: str
    merged_with: Optional[str] = None  # original fact_id if merged
    similarity: float = 0.0


# ---------------------------------------------------------------------------
# Layer 3 — Working Context
# ---------------------------------------------------------------------------

@dataclass
class WorkingContext:
    """Per-(project_id, user_id) live working state. UPSERT-only."""
    project_id: str = ""
    user_id: str = ""
    current_task: str = ""
    recent_decisions: list[str] = field(default_factory=list)   # last 5 titles
    open_questions: list[str] = field(default_factory=list)
    files_in_focus: list[str] = field(default_factory=list)
    last_session_summary: str = ""
    next_steps: str = ""
    blockers: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "user_id": self.user_id,
            "current_task": self.current_task,
            "recent_decisions": self.recent_decisions,
            "open_questions": self.open_questions,
            "files_in_focus": self.files_in_focus,
            "last_session_summary": self.last_session_summary,
            "next_steps": self.next_steps,
            "blockers": self.blockers,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkingContext":
        ctx = cls()
        for k, v in d.items():
            if hasattr(ctx, k):
                setattr(ctx, k, v)
        return ctx

    def as_prompt_block(self) -> str:
        """Formatted markdown for injection into agent system prompt."""
        parts = [f"<!-- XME Context | project={self.project_id} | user={self.user_id} -->"]
        if self.current_task:
            parts.append(f"**Current task**: {self.current_task}")
        if self.last_session_summary:
            parts.append(f"**Last session**: {self.last_session_summary}")
        if self.recent_decisions:
            parts.append("**Recent decisions**:")
            for d in self.recent_decisions[:5]:
                parts.append(f"- {d}")
        if self.next_steps:
            parts.append(f"**Next steps**: {self.next_steps}")
        if self.blockers:
            parts.append("**Blockers**: " + "; ".join(self.blockers))
        if self.open_questions:
            parts.append("**Open questions**:")
            for q in self.open_questions[:3]:
                parts.append(f"- {q}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass
class MemorySearchResult:
    layer: str          # "episodic" | "facts" | "context"
    item_id: str
    score: float
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    highlight: Optional[str] = None


@dataclass
class SearchResults:
    query: str
    project_id: str
    episodic: list[MemorySearchResult] = field(default_factory=list)
    facts: list[MemorySearchResult] = field(default_factory=list)
    context: list[MemorySearchResult] = field(default_factory=list)

    @property
    def all_results(self) -> list[MemorySearchResult]:
        combined = self.episodic + self.facts + self.context
        combined.sort(key=lambda r: r.score, reverse=True)
        return combined


# ---------------------------------------------------------------------------
# Session context (returned by session_start)
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    """Primed context returned at session start."""
    session_id: str
    project_id: str
    user_id: str
    working_context: Optional[WorkingContext]
    recent_facts: list[Fact]
    last_episode_summary: str
    prompt_block: str   # ready-to-inject markdown
