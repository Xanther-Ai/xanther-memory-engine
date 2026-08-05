"""Layer 3 — Working Context (SQLite UPSERT store).

Per-(project_id, user_id) state. UPSERT semantics — always reflects latest.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from xme.models import WorkingContext

logger = logging.getLogger(__name__)


class ContextStore:
    """SQLite-backed working context. Lightweight, always fast."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS working_context (
        project_id TEXT NOT NULL,
        user_id    TEXT NOT NULL,
        data       TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (project_id, user_id)
    );
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._conn:
            return
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ContextStore":
        self.connect()
        return self

    def __exit__(self, *_) -> None:  # type: ignore[override]
        self.close()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(self, ctx: WorkingContext) -> None:
        """Insert or replace the working context for (project_id, user_id)."""
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO working_context (project_id, user_id, data, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id, user_id) DO UPDATE SET
                data = excluded.data,
                updated_at = excluded.updated_at
            """,
            (ctx.project_id, ctx.user_id, json.dumps(ctx.to_dict()), ctx.updated_at),
        )
        self._conn.commit()

    def get(self, project_id: str, user_id: str) -> Optional[WorkingContext]:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT data FROM working_context WHERE project_id=? AND user_id=?",
            (project_id, user_id),
        ).fetchone()
        if row:
            return WorkingContext.from_dict(json.loads(row["data"]))
        return None

    def list_users(self, project_id: str) -> list[str]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT user_id FROM working_context WHERE project_id=? ORDER BY updated_at DESC",
            (project_id,),
        ).fetchall()
        return [r["user_id"] for r in rows]

    def delete(self, project_id: str, user_id: str) -> None:
        assert self._conn is not None
        self._conn.execute(
            "DELETE FROM working_context WHERE project_id=? AND user_id=?",
            (project_id, user_id),
        )
        self._conn.commit()

    def update_from_session(
        self,
        project_id: str,
        user_id: str,
        session_summary: str,
        recent_decisions: list[str],
        next_steps: str = "",
        files_touched: Optional[list[str]] = None,
    ) -> WorkingContext:
        """Convenience: update context from session_end data."""
        from xme.models import _now
        ctx = self.get(project_id, user_id) or WorkingContext(
            project_id=project_id, user_id=user_id
        )
        ctx.last_session_summary = session_summary
        # Prepend new decisions, keep last 5
        all_dec = recent_decisions + [d for d in ctx.recent_decisions if d not in recent_decisions]
        ctx.recent_decisions = all_dec[:5]
        if next_steps:
            ctx.next_steps = next_steps
        if files_touched:
            ctx.files_in_focus = list(dict.fromkeys(files_touched + ctx.files_in_focus))[:10]
        ctx.updated_at = _now()
        self.upsert(ctx)
        return ctx

    def get_prompt_block(self, project_id: str, user_id: str) -> str:
        ctx = self.get(project_id, user_id)
        if ctx:
            return ctx.as_prompt_block()
        return f"<!-- XME: no context yet for project={project_id} user={user_id} -->"
