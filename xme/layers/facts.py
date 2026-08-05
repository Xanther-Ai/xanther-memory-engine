"""Layer 2 — Fact Graph Store (Neo4j with SQLite fallback).

UPSERT semantics: before inserting a new fact, check for duplicates
via vector similarity. If a similar fact exists (cosine > threshold),
merge/update it instead of creating a duplicate.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from xme.models import Confidence, Fact, FactType, MemorySearchResult, UpsertResult

logger = logging.getLogger(__name__)


class FactGraphStore:
    """Neo4j-backed fact graph with UPSERT deduplication.

    Falls back to SQLite when Neo4j is unavailable.
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_auth: tuple[str, str] = ("neo4j", ""),
        sqlite_path: str = ".xanther/xme.db",
        dedup_threshold: float = 0.85,
        embedding_dims: int = 384,
        neo4j_enabled: bool = True,
    ) -> None:
        self._neo4j_uri = neo4j_uri
        self._neo4j_auth = neo4j_auth
        self._sqlite_path = Path(sqlite_path)
        self._threshold = dedup_threshold
        self._dims = embedding_dims
        self._neo4j_enabled = neo4j_enabled

        self._driver: Optional[Any] = None
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._sqlite_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_sqlite_schema()
        if self._neo4j_enabled:
            await self._init_neo4j()

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aenter__(self) -> "FactGraphStore":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:  # type: ignore[override]
        await self.close()

    # ------------------------------------------------------------------
    # UPSERT — the key operation
    # ------------------------------------------------------------------

    async def upsert_fact(
        self,
        fact: Fact,
        embedder: Optional[Any] = None,
    ) -> UpsertResult:
        """Insert or merge a fact based on semantic similarity.

        1. If embedder provided and fact has no embedding: generate one
        2. Search for similar facts in same project
        3. If match >= threshold: update existing fact, return "merged"
        4. Else: create new fact, return "created"
        """
        # Generate embedding if needed
        if embedder is not None and not fact.embedding:
            fact.embedding = embedder.embed(fact.text_for_embedding())

        # Search for duplicates
        if fact.embedding and any(x != 0.0 for x in fact.embedding):
            similar = await self._find_similar(
                fact.embedding, fact.project_id, fact.fact_type, top_k=3
            )
            for candidate_id, similarity in similar:
                if similarity >= self._threshold:
                    merged = await self._merge_fact(candidate_id, fact)
                    return UpsertResult(
                        action="merged",
                        fact_id=merged.fact_id,
                        merged_with=candidate_id,
                        similarity=similarity,
                    )

        # No duplicate — create new
        await self._save_fact(fact)
        return UpsertResult(action="created", fact_id=fact.fact_id)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_fact(self, fact_id: str) -> Optional[Fact]:
        if self._driver:
            return await self._get_neo4j(fact_id)
        return self._get_sqlite(fact_id)

    async def search_facts(
        self,
        query: str,
        project_id: str,
        fact_type: Optional[str] = None,
        limit: int = 20,
        query_embedding: Optional[list[float]] = None,
    ) -> list[MemorySearchResult]:
        if self._driver and query_embedding:
            return await self._search_neo4j_vector(
                query_embedding, project_id, fact_type, limit
            )
        return self._search_sqlite_keyword(query, project_id, fact_type, limit)

    async def list_facts(
        self,
        project_id: str,
        fact_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[Fact]:
        assert self._conn is not None
        q = "SELECT data FROM xme_facts WHERE project_id=? AND status='active'"
        params: list[Any] = [project_id]
        if fact_type:
            q += " AND fact_type=?"
            params.append(fact_type)
        if user_id:
            q += " AND user_id=?"
            params.append(user_id)
        q += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(q, params).fetchall()
        return [Fact.from_dict(json.loads(r["data"])) for r in rows]

    async def get_related(self, fact_id: str, depth: int = 2) -> list[Fact]:
        if self._driver:
            return await self._get_related_neo4j(fact_id, depth)
        return self._get_related_sqlite(fact_id)

    async def delete_fact(self, fact_id: str) -> None:
        """Soft delete — sets status=deleted."""
        from xme.models import _now
        assert self._conn is not None
        self._conn.execute(
            "UPDATE xme_facts SET status='deleted', updated_at=? WHERE fact_id=?",
            (_now(), fact_id),
        )
        self._conn.commit()
        if self._driver:
            await self._delete_neo4j(fact_id)

    async def link_to_code(self, fact_id: str, ast_node_ids: list[str]) -> None:
        """Link a fact to XCE AST nodes (when XCE is present)."""
        if not ast_node_ids:
            return
        assert self._conn is not None
        fact = self._get_sqlite(fact_id)
        if fact:
            fact.code_node_ids = list(set(fact.code_node_ids + ast_node_ids))
            from xme.models import _now
            fact.updated_at = _now()
            self._save_sqlite_only(fact)
        if self._driver:
            await self._link_code_neo4j(fact_id, ast_node_ids)

    async def link_episode_to_facts(
        self, episode_id: str, fact_ids: list[str]
    ) -> None:
        """Link an episode to the facts extracted from it."""
        if not fact_ids:
            return
        if self._driver:
            await self._link_episode_neo4j(episode_id, fact_ids)

    # ------------------------------------------------------------------
    # Graph export (for dashboard + Graphify-compat)
    # ------------------------------------------------------------------

    async def get_graph_data(self, project_id: str) -> dict[str, Any]:
        """Return nodes + edges in vis.js format."""
        facts = await self.list_facts(project_id, limit=500)
        nodes = []
        edges = []
        seen_edges: set[tuple[str, str]] = set()

        for f in facts:
            nodes.append({
                "id": f.fact_id,
                "label": f.title[:40],
                "group": f.fact_type,
                "title": f.content[:200],
                "confidence": f.confidence,
            })
            for rel_id in f.related_fact_ids:
                key = tuple(sorted([f.fact_id, rel_id]))
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append({
                        "from": f.fact_id,
                        "to": rel_id,
                        "label": "RELATED_TO",
                    })

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # SQLite backend
    # ------------------------------------------------------------------

    def _init_sqlite_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS xme_facts (
            fact_id    TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            user_id    TEXT NOT NULL,
            fact_type  TEXT NOT NULL,
            title      TEXT NOT NULL DEFAULT '',
            status     TEXT NOT NULL DEFAULT 'active',
            updated_at TEXT NOT NULL,
            data       TEXT NOT NULL,
            embedding  BLOB
        );
        CREATE INDEX IF NOT EXISTS fct_project_idx ON xme_facts(project_id);
        CREATE INDEX IF NOT EXISTS fct_type_idx    ON xme_facts(project_id, fact_type);
        CREATE INDEX IF NOT EXISTS fct_user_idx    ON xme_facts(user_id);
        CREATE INDEX IF NOT EXISTS fct_status_idx  ON xme_facts(status);
        """)
        self._conn.commit()

    async def _save_fact(self, fact: Fact) -> None:
        self._save_sqlite_only(fact)
        if self._driver:
            await self._upsert_neo4j(fact)

    def _save_sqlite_only(self, fact: Fact) -> None:
        assert self._conn is not None
        embedding_bytes = (
            bytes(json.dumps(fact.embedding).encode()) if fact.embedding else None
        )
        self._conn.execute(
            """
            INSERT INTO xme_facts
                (fact_id, project_id, user_id, fact_type, title, status,
                 updated_at, data, embedding)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(fact_id) DO UPDATE SET
                title=excluded.title,
                status=excluded.status,
                updated_at=excluded.updated_at,
                data=excluded.data,
                embedding=excluded.embedding
            """,
            (
                fact.fact_id, fact.project_id, fact.user_id, fact.fact_type,
                fact.title, fact.status, fact.updated_at,
                json.dumps(fact.to_dict()), embedding_bytes,
            ),
        )
        self._conn.commit()

    def _get_sqlite(self, fact_id: str) -> Optional[Fact]:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT data FROM xme_facts WHERE fact_id=?", (fact_id,)
        ).fetchone()
        return Fact.from_dict(json.loads(row["data"])) if row else None

    async def _find_similar(
        self,
        embedding: list[float],
        project_id: str,
        fact_type: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Find fact_ids with similar embeddings. Returns (fact_id, similarity) pairs."""
        if self._driver:
            return await self._vector_search_neo4j(embedding, project_id, fact_type, top_k)
        return self._vector_search_sqlite(embedding, project_id, fact_type, top_k)

    def _vector_search_sqlite(
        self,
        embedding: list[float],
        project_id: str,
        fact_type: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Brute-force cosine similarity over SQLite (adequate for < 10K facts)."""
        from xme.extraction.embedder import LocalEmbedder
        assert self._conn is not None

        rows = self._conn.execute(
            "SELECT fact_id, embedding FROM xme_facts "
            "WHERE project_id=? AND fact_type=? AND status='active' AND embedding IS NOT NULL",
            (project_id, fact_type),
        ).fetchall()

        results: list[tuple[str, float]] = []
        for row in rows:
            try:
                stored = json.loads(row["embedding"].decode())
                sim = LocalEmbedder.cosine_similarity(embedding, stored)
                results.append((row["fact_id"], sim))
            except Exception:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def _merge_fact(self, existing_id: str, new_fact: Fact) -> Fact:
        """Merge new_fact info into existing fact."""
        from xme.models import _now
        existing = self._get_sqlite(existing_id)
        if existing is None:
            await self._save_fact(new_fact)
            return new_fact

        # Merge: extend content, update metadata, keep existing id
        if new_fact.content and new_fact.content not in existing.content:
            existing.content = f"{existing.content}\n\nUpdate: {new_fact.content}"
        # Merge metadata dicts
        existing.metadata.update(new_fact.metadata)
        # Update source linkage
        if new_fact.source_episode_id and new_fact.source_episode_id not in str(existing.metadata):
            existing.related_fact_ids = list(set(
                existing.related_fact_ids + [new_fact.fact_id]
            ))
        existing.updated_at = _now()
        # Use new embedding (more recent context)
        if new_fact.embedding:
            existing.embedding = new_fact.embedding
        self._save_sqlite_only(existing)
        if self._driver:
            await self._upsert_neo4j(existing)
        return existing

    def _search_sqlite_keyword(
        self,
        query: str,
        project_id: str,
        fact_type: Optional[str],
        limit: int,
    ) -> list[MemorySearchResult]:
        assert self._conn is not None
        terms = query.lower().split()
        if not terms:
            return []
        like = " OR ".join(["LOWER(data) LIKE ?" for _ in terms])
        params: list[Any] = [f"%{t}%" for t in terms]
        base = f"SELECT fact_id, data FROM xme_facts WHERE ({like}) AND project_id=? AND status='active'"
        params.append(project_id)
        if fact_type:
            base += " AND fact_type=?"
            params.append(fact_type)
        base += f" ORDER BY updated_at DESC LIMIT {limit}"
        rows = self._conn.execute(base, params).fetchall()
        results = []
        for i, row in enumerate(rows):
            d = json.loads(row["data"])
            results.append(MemorySearchResult(
                layer="facts",
                item_id=row["fact_id"],
                score=1.0 - i * 0.05,
                summary=d.get("title", ""),
                data=d,
            ))
        return results

    def _get_related_sqlite(self, fact_id: str) -> list[Fact]:
        f = self._get_sqlite(fact_id)
        if not f or not f.related_fact_ids:
            return []
        results = []
        for rid in f.related_fact_ids[:10]:
            rel = self._get_sqlite(rid)
            if rel:
                results.append(rel)
        return results

    # ------------------------------------------------------------------
    # Neo4j backend
    # ------------------------------------------------------------------

    async def _init_neo4j(self) -> None:
        try:
            from neo4j import AsyncGraphDatabase
            self._driver = AsyncGraphDatabase.driver(
                self._neo4j_uri, auth=self._neo4j_auth
            )
            async with self._driver.session() as s:
                # Schema for Fact nodes
                for stmt in [
                    "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE",
                    "CREATE INDEX fact_project_idx IF NOT EXISTS FOR (f:Fact) ON (f.project_id)",
                    "CREATE INDEX fact_type_idx IF NOT EXISTS FOR (f:Fact) ON (f.fact_type)",
                    (
                        "CREATE VECTOR INDEX fact_embedding_idx IF NOT EXISTS "
                        "FOR (f:Fact) ON (f.embedding) "
                        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {self._dims}, "
                        "`vector.similarity_function`: 'cosine'}}}}"
                    ),
                ]:
                    await s.run(stmt)
            logger.debug("FactGraphStore Neo4j connected")
        except Exception as e:
            logger.warning("Neo4j unavailable: %s — using SQLite for facts", e)
            self._driver = None

    async def _upsert_neo4j(self, fact: Fact) -> None:
        assert self._driver is not None
        async with self._driver.session() as s:
            await s.run(
                f"""
                MERGE (f:Fact:{fact.fact_type.capitalize()} {{fact_id: $fid}})
                SET f.project_id = $project_id,
                    f.user_id = $user_id,
                    f.fact_type = $fact_type,
                    f.title = $title,
                    f.content = $content,
                    f.confidence = $confidence,
                    f.status = $status,
                    f.updated_at = $updated_at,
                    f.embedding = $embedding
                """,
                {
                    "fid": fact.fact_id,
                    "project_id": fact.project_id,
                    "user_id": fact.user_id,
                    "fact_type": fact.fact_type,
                    "title": fact.title,
                    "content": fact.content,
                    "confidence": fact.confidence,
                    "status": fact.status,
                    "updated_at": fact.updated_at,
                    "embedding": fact.embedding if fact.embedding else None,
                },
            )

    async def _get_neo4j(self, fact_id: str) -> Optional[Fact]:
        assert self._driver is not None
        async with self._driver.session() as s:
            result = await s.run("MATCH (f:Fact {fact_id: $fid}) RETURN properties(f) AS p", {"fid": fact_id})
            record = await result.single()
            if record:
                return Fact.from_dict(dict(record["p"]))
        return None

    async def _vector_search_neo4j(
        self,
        embedding: list[float],
        project_id: str,
        fact_type: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        assert self._driver is not None
        try:
            async with self._driver.session() as s:
                result = await s.run(
                    """
                    CALL db.index.vector.queryNodes('fact_embedding_idx', $k, $emb)
                    YIELD node, score
                    WHERE node.project_id = $project_id
                      AND node.fact_type = $fact_type
                      AND node.status = 'active'
                    RETURN node.fact_id AS fact_id, score
                    """,
                    {"k": top_k, "emb": embedding,
                     "project_id": project_id, "fact_type": fact_type},
                )
                return [(r["fact_id"], r["score"]) async for r in result]
        except Exception as e:
            logger.warning("Neo4j vector search failed: %s", e)
            return []

    async def _search_neo4j_vector(
        self,
        embedding: list[float],
        project_id: str,
        fact_type: Optional[str],
        limit: int,
    ) -> list[MemorySearchResult]:
        assert self._driver is not None
        try:
            async with self._driver.session() as s:
                where = "node.project_id = $project_id AND node.status = 'active'"
                params: dict[str, Any] = {"k": limit, "emb": embedding, "project_id": project_id}
                if fact_type:
                    where += " AND node.fact_type = $fact_type"
                    params["fact_type"] = fact_type
                result = await s.run(
                    f"""
                    CALL db.index.vector.queryNodes('fact_embedding_idx', $k, $emb)
                    YIELD node, score
                    WHERE {where}
                    RETURN node.fact_id AS fact_id, node.title AS title,
                           node.content AS content, properties(node) AS props, score
                    LIMIT $k
                    """,
                    params,
                )
                results = []
                async for r in result:
                    results.append(MemorySearchResult(
                        layer="facts",
                        item_id=r["fact_id"],
                        score=r["score"],
                        summary=r["title"],
                        data=dict(r["props"]),
                    ))
                return results
        except Exception as e:
            logger.warning("Neo4j search failed: %s", e)
            return []

    async def _get_related_neo4j(self, fact_id: str, depth: int) -> list[Fact]:
        assert self._driver is not None
        try:
            async with self._driver.session() as s:
                result = await s.run(
                    f"""
                    MATCH (f:Fact {{fact_id: $fid}})-[*1..{min(depth,3)}]-(related:Fact)
                    WHERE related.status = 'active'
                    RETURN DISTINCT properties(related) AS p
                    LIMIT 20
                    """,
                    {"fid": fact_id},
                )
                return [Fact.from_dict(dict(r["p"])) async for r in result]
        except Exception as e:
            logger.warning("Neo4j get_related failed: %s", e)
            return []

    async def _delete_neo4j(self, fact_id: str) -> None:
        assert self._driver is not None
        from xme.models import _now
        async with self._driver.session() as s:
            await s.run(
                "MATCH (f:Fact {fact_id: $fid}) SET f.status='deleted', f.updated_at=$ts",
                {"fid": fact_id, "ts": _now()},
            )

    async def _link_code_neo4j(self, fact_id: str, ast_node_ids: list[str]) -> None:
        assert self._driver is not None
        async with self._driver.session() as s:
            for nid in ast_node_ids:
                await s.run(
                    """
                    MATCH (f:Fact {fact_id: $fid})
                    MATCH (a:ASTNode {id: $nid})
                    MERGE (f)-[:REFERENCES_CODE]->(a)
                    """,
                    {"fid": fact_id, "nid": nid},
                )

    async def _link_episode_neo4j(self, episode_id: str, fact_ids: list[str]) -> None:
        assert self._driver is not None
        async with self._driver.session() as s:
            for fid in fact_ids:
                await s.run(
                    """
                    MATCH (f:Fact {fact_id: $fid})
                    MERGE (ep:Episode {episode_id: $eid})
                    MERGE (f)-[:EXTRACTED_FROM]->(ep)
                    """,
                    {"fid": fid, "eid": episode_id},
                )
