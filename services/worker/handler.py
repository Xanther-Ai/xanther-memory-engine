"""
XME Worker (Xanther Memory Engine)

Simplified architecture:
1. Reads unprocessed raw messages from PostgreSQL
2. Extracts structured knowledge (decisions, bugs, insights) via LLM
3. Writes distilled memory nodes to Neo4j (max 500 per repo)
4. Marks messages as processed

Runs as a Docker container on the same EC2 as XCE + Neo4j + PostgreSQL.
"""

import os
import sys
import json
import time
import logging
import signal
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass

import psycopg2
import psycopg2.extras
from neo4j import GraphDatabase
import httpx

from git_linker import GitLinker, SessionArtifactGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [XME] %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


# --- Configuration ---

class Config:
    # PostgreSQL (TimescaleDB on same EC2)
    PG_HOST = os.getenv("PG_HOST", "xme-postgres")
    PG_PORT = int(os.getenv("PG_PORT", "5432"))
    PG_DB = os.getenv("PG_DB", "xce_memory")
    PG_USER = os.getenv("PG_USER", "xce_memory")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "xme_prod_password")

    # Neo4j (existing container on same network)
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://xce-neo4j:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "xce_prod_password")

    # LLM for extraction (use XCE's existing OpenRouter key)
    LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")

    # Processing
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
    EXTRACTION_INTERVAL = int(os.getenv("EXTRACTION_INTERVAL", "300"))  # 5 minutes
    MAX_MEMORY_NODES_PER_REPO = int(os.getenv("MAX_MEMORY_NODES", "500"))


# --- Extraction Prompt ---

EXTRACTION_PROMPT = """You are analyzing a coding session conversation. Extract ONLY the following types of knowledge:

1. **Decisions** - Architectural or design decisions made (and why)
2. **Bugs** - Bugs discovered during the session
3. **Insights** - Code patterns, performance observations, or learnings
4. **Questions** - Important questions that were answered with useful context

For each item, provide:
- `kind`: one of "decision", "bug", "insight", "question"
- `summary`: 1-2 sentence summary
- `reasoning`: why this matters (1 sentence, optional)
- `confidence`: 0.0-1.0 how confident you are this is important
- `references`: list of file paths or function/class names mentioned

ONLY extract items that are genuinely useful for future sessions. Skip trivial edits, typo fixes, or routine operations.

Return JSON array. If nothing worth extracting, return empty array [].

Messages:
{messages}"""


# --- PostgreSQL Client ---

class PGClient:
    def __init__(self, config: Config):
        self.config = config
        self.conn = None

    def connect(self):
        self.conn = psycopg2.connect(
            host=self.config.PG_HOST,
            port=self.config.PG_PORT,
            database=self.config.PG_DB,
            user=self.config.PG_USER,
            password=self.config.PG_PASSWORD
        )
        self.conn.autocommit = True
        logger.info(f"Connected to PostgreSQL at {self.config.PG_HOST}")

    def close(self):
        if self.conn:
            self.conn.close()

    def get_unprocessed_messages(self, limit: int = 20) -> list:
        """Get batch of unprocessed raw messages grouped by session."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, session_id, user_id, repo_id, role, content, tool_calls, created_at
                FROM raw_messages
                WHERE processed = FALSE
                ORDER BY created_at ASC
                LIMIT %s
            """, (limit,))
            return cur.fetchall()

    def mark_processed(self, message_ids: list):
        """Mark messages as processed."""
        if not message_ids:
            return
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE raw_messages SET processed = TRUE
                WHERE id = ANY(%s::uuid[])
            """, ([str(mid) for mid in message_ids],))

    def store_extracted_memory(self, memory: dict):
        """Store extracted memory in PostgreSQL."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO extracted_memories
                (session_id, user_id, repo_id, kind, summary, reasoning, confidence, priority, refs, source_message_ids)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                memory["session_id"],
                memory["user_id"],
                memory["repo_id"],
                memory["kind"],
                memory["summary"],
                memory.get("reasoning"),
                memory.get("confidence", 0.8),
                memory.get("priority", 2),
                json.dumps(memory.get("references", [])),
                memory.get("source_message_ids", [])
            ))
            return cur.fetchone()[0]

    def get_unwritten_memories(self, limit: int = 50) -> list:
        """Get extracted memories not yet written to Neo4j."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, session_id, user_id, repo_id, kind, summary, reasoning,
                       confidence, priority, refs, created_at
                FROM extracted_memories
                WHERE written_to_graph = FALSE
                ORDER BY priority ASC, created_at ASC
                LIMIT %s
            """, (limit,))
            return cur.fetchall()

    def mark_written_to_graph(self, memory_ids: list):
        """Mark memories as written to Neo4j."""
        if not memory_ids:
            return
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE extracted_memories SET written_to_graph = TRUE
                WHERE id = ANY(%s::uuid[])
            """, ([str(mid) for mid in memory_ids],))

    def insert_raw_message(self, session_id: str, user_id: str, repo_id: str,
                           role: str, content: str, tool_calls: dict = None):
        """Insert a raw message (called by XCE MCP server)."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO raw_messages (session_id, user_id, repo_id, role, content, tool_calls)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (session_id, user_id, repo_id, role, content,
                  json.dumps(tool_calls) if tool_calls else None))
            return cur.fetchone()[0]


# --- Neo4j Client (with eviction) ---

class Neo4jMemoryClient:
    def __init__(self, config: Config):
        self.config = config
        self.driver = None
        self.max_nodes = config.MAX_MEMORY_NODES_PER_REPO

    def connect(self):
        self.driver = GraphDatabase.driver(
            self.config.NEO4J_URI,
            auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD)
        )
        logger.info(f"Connected to Neo4j at {self.config.NEO4J_URI}")

    def close(self):
        if self.driver:
            self.driver.close()

    def get_memory_count(self, repo_id: str) -> int:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (m:Memory {repo_id: $repo_id}) RETURN count(m) as cnt",
                {"repo_id": repo_id}
            )
            record = result.single()
            return record["cnt"] if record else 0

    def evict_if_needed(self, repo_id: str):
        """Evict old/low-confidence memories if at cap."""
        count = self.get_memory_count(repo_id)
        if count < self.max_nodes:
            return

        to_free = count - self.max_nodes + 10
        with self.driver.session() as session:
            # Evict lowest confidence + oldest first
            session.run("""
                MATCH (m:Memory {repo_id: $repo_id})
                WITH m ORDER BY m.confidence ASC, m.created_at ASC
                LIMIT $limit
                DETACH DELETE m
            """, {"repo_id": repo_id, "limit": to_free})
            logger.info(f"Evicted {to_free} memories for repo {repo_id}")

    def write_memory(self, memory: dict) -> bool:
        """Write a distilled memory node to Neo4j."""
        repo_id = memory["repo_id"]

        # Check cap
        self.evict_if_needed(repo_id)

        try:
            with self.driver.session() as session:
                # Create Memory node
                session.run("""
                    CREATE (m:Memory {
                        id: $id,
                        session_id: $session_id,
                        repo_id: $repo_id,
                        kind: $kind,
                        summary: $summary,
                        reasoning: $reasoning,
                        confidence: $confidence,
                        priority: $priority,
                        created_at: datetime($created_at)
                    })
                """, {
                    "id": str(memory["id"]),
                    "session_id": memory["session_id"],
                    "repo_id": repo_id,
                    "kind": memory["kind"],
                    "summary": memory["summary"],
                    "reasoning": memory.get("reasoning", ""),
                    "confidence": memory.get("confidence", 0.8),
                    "priority": memory.get("priority", 2),
                    "created_at": memory["created_at"].isoformat() if hasattr(memory["created_at"], 'isoformat') else str(memory["created_at"])
                })

                # Link to code nodes if refs exist
                refs = memory.get("refs")
                if refs:
                    if isinstance(refs, str):
                        refs = json.loads(refs)
                    for ref in refs:
                        ref_name = ref if isinstance(ref, str) else ref.get("name", ref.get("node_id", ""))
                        if ref_name:
                            # Try to find matching ASTNode by name or filepath
                            session.run("""
                                MATCH (m:Memory {id: $memory_id})
                                OPTIONAL MATCH (c:ASTNode)
                                WHERE c.name = $ref_name OR c.filepath CONTAINS $ref_name
                                WITH m, c LIMIT 1
                                WHERE c IS NOT NULL
                                MERGE (m)-[:REFS]->(c)
                            """, {"memory_id": str(memory["id"]), "ref_name": ref_name})

            return True
        except Exception as e:
            logger.error(f"Failed to write memory to Neo4j: {e}")
            return False


# --- LLM Extraction ---

class LLMExtractor:
    def __init__(self, config: Config):
        self.config = config
        self.client = httpx.Client(timeout=60.0)

    def extract_knowledge(self, messages: list) -> list:
        """Use LLM to extract structured knowledge from raw messages."""
        if not self.config.LLM_API_KEY:
            logger.warning("No LLM API key configured, skipping extraction")
            return []

        # Format messages for the prompt
        formatted = "\n".join([
            f"[{m['role']}] {m['content'][:500]}"
            for m in messages
        ])

        prompt = EXTRACTION_PROMPT.format(messages=formatted)

        try:
            response = self.client.post(
                self.config.LLM_API_URL,
                headers={
                    "Authorization": f"Bearer {self.config.LLM_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config.LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 2000
                }
            )

            if response.status_code != 200:
                logger.error(f"LLM API error: {response.status_code}")
                return []

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            extracted = json.loads(content.strip())
            return extracted if isinstance(extracted, list) else []

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []


# --- Main Worker ---

class XMEWorker:
    def __init__(self):
        self.config = Config()
        self.pg = PGClient(self.config)
        self.neo4j = Neo4jMemoryClient(self.config)
        self.llm = LLMExtractor(self.config)
        self.running = False

    def connect(self):
        self.pg.connect()
        self.neo4j.connect()
        logger.info("XME Worker connected to all services")

    def close(self):
        self.pg.close()
        self.neo4j.close()

    def process_batch(self):
        """Main processing cycle: extract from raw messages, write to graph, link commits."""

        # 1. Get unprocessed messages
        messages = self.pg.get_unprocessed_messages(self.config.BATCH_SIZE)
        if not messages:
            return

        logger.info(f"Processing {len(messages)} raw messages")

        # 2. Group by session
        sessions = {}
        for msg in messages:
            sid = msg["session_id"]
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(msg)

        # 3. Extract knowledge from each session batch
        for session_id, session_messages in sessions.items():
            extracted = self.llm.extract_knowledge(session_messages)

            if extracted:
                logger.info(f"Extracted {len(extracted)} items from session {session_id[:8]}")

                # Get metadata from first message
                first_msg = session_messages[0]
                message_ids = [m["id"] for m in session_messages]

                for item in extracted:
                    # Only store if confidence is high enough
                    if item.get("confidence", 0.8) < 0.6:
                        continue

                    memory = {
                        "session_id": session_id,
                        "user_id": first_msg["user_id"],
                        "repo_id": first_msg["repo_id"],
                        "kind": item.get("kind", "insight"),
                        "summary": item.get("summary", ""),
                        "reasoning": item.get("reasoning"),
                        "confidence": item.get("confidence", 0.8),
                        "priority": self._infer_priority(item.get("kind", "insight")),
                        "references": item.get("references", []),
                        "source_message_ids": message_ids
                    }

                    self.pg.store_extracted_memory(memory)

            # Mark messages as processed
            msg_ids = [m["id"] for m in session_messages]
            self.pg.mark_processed(msg_ids)

        # 4. Write unwritten memories to Neo4j
        unwritten = self.pg.get_unwritten_memories(50)
        written_ids = []

        for mem in unwritten:
            if self.neo4j.write_memory(mem):
                written_ids.append(mem["id"])

        if written_ids:
            self.pg.mark_written_to_graph(written_ids)
            logger.info(f"Wrote {len(written_ids)} memories to Neo4j")

        # 5. Phase 2: Link git commits to sessions
        self._link_git_commits()

    def _infer_priority(self, kind: str) -> int:
        return {"bug": 0, "decision": 1, "question": 1, "insight": 2, "pattern": 3}.get(kind, 2)

    def _link_git_commits(self):
        """Phase 2: Link recent git commits to sessions and code nodes."""
        repo_path = os.getenv("REPO_PATH", "/repos/current")
        if not os.path.exists(repo_path):
            return

        try:
            linker = GitLinker(
                neo4j_driver=self.neo4j.driver,
                pg_conn=self.pg.conn,
                repo_path=repo_path,
                repo_id=os.getenv("REPO_ID", "")
            )
            stats = linker.process_recent_commits(since_hours=1)
            if stats["commits"] > 0:
                logger.info(f"Git linking: {stats}")
        except Exception as e:
            logger.error(f"Git linking failed: {e}")

    def _generate_session_artifacts(self):
        """Generate MD summaries for completed sessions."""
        try:
            artifact_gen = SessionArtifactGenerator(
                pg_conn=self.pg.conn,
                output_dir=os.getenv("ARTIFACTS_DIR", "memory/sessions")
            )

            # Find sessions that ended > 30 min ago and don't have artifacts yet
            with self.pg.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT DISTINCT session_id
                    FROM extracted_memories
                    WHERE written_to_graph = TRUE
                    AND created_at < NOW() - INTERVAL '30 minutes'
                    AND session_id NOT IN (
                        SELECT session_id FROM extracted_memories
                        WHERE created_at > NOW() - INTERVAL '30 minutes'
                    )
                    LIMIT 5
                """)
                sessions = cur.fetchall()

            for row in sessions:
                artifact_gen.generate_session_summary(row["session_id"])

        except Exception as e:
            logger.error(f"Session artifact generation failed: {e}")

    def run(self):
        """Main loop."""
        self.running = True
        logger.info("Starting XME Worker...")
        self.connect()

        def shutdown(sig, frame):
            logger.info("Shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        while self.running:
            try:
                self.process_batch()
            except Exception as e:
                logger.error(f"Error in processing cycle: {e}")

            time.sleep(self.config.EXTRACTION_INTERVAL)

        self.close()
        logger.info("XME Worker stopped")


if __name__ == "__main__":
    worker = XMEWorker()
    worker.run()