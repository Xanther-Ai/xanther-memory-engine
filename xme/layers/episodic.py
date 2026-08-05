"""Layer 1 — Episodic Store.

Primary backend: OpenSearch (full-text + k-NN semantic search).
Fallback backend: SQLite FTS5 (no external services needed).

Both implement the same interface so callers are backend-agnostic.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from xme.models import Episode, MemorySearchResult, Turn

logger = logging.getLogger(__name__)

_INDEX_PREFIX = "xme_episodes"


class EpisodicStore:
    """OpenSearch primary, SQLite FTS5 fallback."""

    def __init__(
        self,
        opensearch_url: str = "http://localhost:9200",
        sqlite_path: str = ".xanther/xme.db",
        opensearch_enabled: bool = True,
        embedding_dims: int = 384,
    ) -> None:
        self._url = opensearch_url
        self._sqlite_path = Path(sqlite_path)
        self._opensearch_enabled = opensearch_enabled
        self._embedding_dims = embedding_dims

        # Active session buffers: session_id → list[Turn]
        self._buffers: dict[str, list[Turn]] = {}

        # SQLite connection (always available)
        self._conn: Optional[sqlite3.Connection] = None
        self._os_client: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._sqlite_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_sqlite_schema()
        if self._opensearch_enabled:
            self._init_opensearch()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._os_client:
            self._os_client.close()
            self._os_client = None

    def __enter__(self) -> "EpisodicStore":
        self.connect()
        return self

    def __exit__(self, *_) -> None:  # type: ignore[override]
        self.close()

    # ------------------------------------------------------------------
    # Per-turn buffering (non-blocking during session)
    # ------------------------------------------------------------------

    def append_turn(self, session_id: str, turn: Turn) -> None:
        """Buffer a turn in memory. Non-blocking."""
        if session_id not in self._buffers:
            self._buffers[session_id] = []
        self._buffers[session_id].append(turn)

    def get_buffered_turns(self, session_id: str) -> list[Turn]:
        return list(self._buffers.get(session_id, []))

    # ------------------------------------------------------------------
    # Session end — persist episode
    # ------------------------------------------------------------------

    async def save_episode(self, episode: Episode) -> None:
        """Persist a completed episode to both SQLite and OpenSearch."""
        # Merge buffered turns if any
        buffered = self._buffers.pop(episode.session_id, [])
        if buffered:
            episode.turns = episode.turns + buffered

        self._save_sqlite(episode)

        if self._opensearch_enabled and self._os_client is not None:
            await self._index_opensearch(episode)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        project_id: str,
        user_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 20,
        query_embedding: Optional[list[float]] = None,
    ) -> list[MemorySearchResult]:
        if self._opensearch_enabled and self._os_client is not None:
            return await self._search_opensearch(
                query, project_id, user_id, date_from, date_to, limit, query_embedding
            )
        return self._search_sqlite(query, project_id, user_id, limit)

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        return self._get_sqlite(episode_id)

    async def list_episodes(
        self,
        project_id: str,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[Episode]:
        assert self._conn is not None
        q = "SELECT data FROM xme_episodes WHERE project_id=?"
        params: list[Any] = [project_id]
        if user_id:
            q += " AND user_id=?"
            params.append(user_id)
        q += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(q, params).fetchall()
        return [Episode.from_dict(json.loads(r["data"])) for r in rows]

    # ------------------------------------------------------------------
    # SQLite backend
    # ------------------------------------------------------------------

    def _init_sqlite_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS xme_episodes (
            episode_id  TEXT PRIMARY KEY,
            project_id  TEXT NOT NULL,
            user_id     TEXT NOT NULL,
            session_id  TEXT NOT NULL,
            started_at  TEXT NOT NULL,
            ended_at    TEXT,
            summary     TEXT NOT NULL DEFAULT '',
            outcome     TEXT NOT NULL DEFAULT 'unknown',
            data        TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ep_project_idx ON xme_episodes(project_id);
        CREATE INDEX IF NOT EXISTS ep_user_idx    ON xme_episodes(project_id, user_id);
        CREATE INDEX IF NOT EXISTS ep_date_idx    ON xme_episodes(started_at);

        CREATE VIRTUAL TABLE IF NOT EXISTS xme_episodes_fts
        USING fts5(
            episode_id UNINDEXED,
            project_id UNINDEXED,
            transcript,
            summary,
            content='xme_episodes',
            content_rowid='rowid'
        );

        CREATE TRIGGER IF NOT EXISTS ep_fts_insert
        AFTER INSERT ON xme_episodes BEGIN
            INSERT INTO xme_episodes_fts(rowid, episode_id, project_id, transcript, summary)
            VALUES (new.rowid, new.episode_id, new.project_id,
                    json_extract(new.data, '$.full_transcript'), new.summary);
        END;
        """)
        self._conn.commit()

    def _save_sqlite(self, ep: Episode) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO xme_episodes
                (episode_id, project_id, user_id, session_id, started_at,
                 ended_at, summary, outcome, data)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(episode_id) DO UPDATE SET
                ended_at=excluded.ended_at,
                summary=excluded.summary,
                outcome=excluded.outcome,
                data=excluded.data
            """,
            (
                ep.episode_id, ep.project_id, ep.user_id, ep.session_id,
                ep.started_at, ep.ended_at, ep.summary, ep.outcome,
                json.dumps(ep.to_dict()),
            ),
        )
        self._conn.commit()

    def _get_sqlite(self, episode_id: str) -> Optional[Episode]:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT data FROM xme_episodes WHERE episode_id=?", (episode_id,)
        ).fetchone()
        return Episode.from_dict(json.loads(row["data"])) if row else None

    def _search_sqlite(
        self,
        query: str,
        project_id: str,
        user_id: Optional[str],
        limit: int,
    ) -> list[MemorySearchResult]:
        assert self._conn is not None
        try:
            q = """
            SELECT e.episode_id, e.summary, e.data,
                   snippet(xme_episodes_fts, 2, '<b>', '</b>', '...', 20) AS hl
            FROM xme_episodes_fts
            JOIN xme_episodes e ON xme_episodes_fts.rowid = e.rowid
            WHERE xme_episodes_fts MATCH ?
              AND xme_episodes_fts.project_id = ?
            """
            params: list[Any] = [_fts_query(query), project_id]
            if user_id:
                q += " AND e.user_id = ?"
                params.append(user_id)
            q += f" LIMIT {limit}"
            rows = self._conn.execute(q, params).fetchall()
        except sqlite3.OperationalError:
            # FTS not populated yet — fall back to LIKE
            rows = []

        results = []
        for i, row in enumerate(rows):
            results.append(MemorySearchResult(
                layer="episodic",
                item_id=row["episode_id"],
                score=1.0 - i * 0.05,
                summary=row["summary"] or "",
                data=json.loads(row["data"]),
                highlight=row["hl"] if "hl" in row.keys() else None,
            ))
        return results

    # ------------------------------------------------------------------
    # OpenSearch backend
    # ------------------------------------------------------------------

    def _init_opensearch(self) -> None:
        try:
            from opensearchpy import AsyncOpenSearch
            self._os_client = AsyncOpenSearch(hosts=[self._url], timeout=5)
        except ImportError:
            logger.warning(
                "opensearch-py not installed. Falling back to SQLite. "
                "Install with: pip install opensearch-py"
            )
            self._os_client = None
        except Exception as e:
            logger.warning("OpenSearch init failed: %s. Using SQLite fallback.", e)
            self._os_client = None

    def _index_name(self, project_id: str) -> str:
        safe = project_id.lower().replace("/", "-").replace(":", "-")
        return f"{_INDEX_PREFIX}_{safe}"

    async def _ensure_index(self, project_id: str) -> None:
        assert self._os_client is not None
        name = self._index_name(project_id)
        try:
            exists = await self._os_client.indices.exists(index=name)
            if not exists:
                await self._os_client.indices.create(index=name, body={
                    "mappings": {
                        "properties": {
                            "episode_id":       {"type": "keyword"},
                            "project_id":       {"type": "keyword"},
                            "user_id":          {"type": "keyword"},
                            "session_id":       {"type": "keyword"},
                            "started_at":       {"type": "date"},
                            "ended_at":         {"type": "date"},
                            "full_transcript":  {"type": "text", "analyzer": "english"},
                            "summary":          {"type": "text", "analyzer": "english"},
                            "outcome":          {"type": "keyword"},
                            "tags":             {"type": "keyword"},
                            "fact_ids":         {"type": "keyword"},
                            "turn_count":       {"type": "integer"},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": self._embedding_dims,
                                "method": {"name": "hnsw", "engine": "lucene"},
                            },
                        }
                    },
                    "settings": {"index": {"knn": True}},
                })
        except Exception as e:
            logger.warning("OpenSearch index create failed: %s", e)

    async def _index_opensearch(self, ep: Episode) -> None:
        assert self._os_client is not None
        try:
            await self._ensure_index(ep.project_id)
            doc = ep.to_dict()
            doc.pop("turns", None)  # don't store raw turns in OS (too large)
            await self._os_client.index(
                index=self._index_name(ep.project_id),
                id=ep.episode_id,
                body=doc,
                refresh="wait_for",
            )
        except Exception as e:
            logger.warning("OpenSearch index failed: %s", e)

    async def _search_opensearch(
        self,
        query: str,
        project_id: str,
        user_id: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        limit: int,
        query_embedding: Optional[list[float]],
    ) -> list[MemorySearchResult]:
        assert self._os_client is not None
        must: list[dict] = [{"match": {"project_id": project_id}}]
        if user_id:
            must.append({"match": {"user_id": user_id}})
        if date_from or date_to:
            date_range: dict[str, Any] = {}
            if date_from:
                date_range["gte"] = date_from
            if date_to:
                date_range["lte"] = date_to
            must.append({"range": {"started_at": date_range}})

        if query_embedding and any(x != 0.0 for x in query_embedding):
            os_query: dict[str, Any] = {
                "query": {"bool": {"must": must}},
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": limit,
                    }
                },
                "size": limit,
            }
        else:
            should = [
                {"match": {"full_transcript": {"query": query, "boost": 1.0}}},
                {"match": {"summary": {"query": query, "boost": 2.0}}},
            ]
            os_query = {
                "query": {"bool": {"must": must, "should": should}},
                "highlight": {
                    "fields": {
                        "full_transcript": {"number_of_fragments": 2, "fragment_size": 150},
                        "summary": {},
                    }
                },
                "size": limit,
            }

        try:
            resp = await self._os_client.search(
                index=self._index_name(project_id), body=os_query
            )
            hits = resp["hits"]["hits"]
        except Exception as e:
            logger.warning("OpenSearch search failed: %s, falling back to SQLite", e)
            return self._search_sqlite(query, project_id, user_id, limit)

        results = []
        for hit in hits:
            src = hit["_source"]
            highlights = hit.get("highlight", {})
            hl_parts = highlights.get("summary", []) + highlights.get("full_transcript", [])
            results.append(MemorySearchResult(
                layer="episodic",
                item_id=src.get("episode_id", hit["_id"]),
                score=hit.get("_score", 0.0),
                summary=src.get("summary", ""),
                data=src,
                highlight=" ... ".join(hl_parts[:2]) if hl_parts else None,
            ))
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fts_query(text: str) -> str:
    """Convert plain text to SQLite FTS5 query (quote individual terms)."""
    terms = [t for t in text.split() if t]
    if not terms:
        return '""'
    return " ".join(f'"{t}"' for t in terms)
