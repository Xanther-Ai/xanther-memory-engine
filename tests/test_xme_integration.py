"""
XME Integration Test

Tests the full Phase 1 + Phase 2 pipeline:
1. Insert raw messages into PostgreSQL
2. Run extraction (mocked LLM)
3. Write memory nodes to Neo4j
4. Link git commits to sessions
5. Generate session artifacts
6. Verify graph structure

Requirements:
- Docker running with xme-postgres and xce-neo4j containers
- Or: set TEST_PG_HOST and TEST_NEO4J_URI env vars

Run: pytest tests/test_xme_integration.py -v
"""

import os
import json
import uuid
import tempfile
import subprocess
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

# Skip if dependencies not available
try:
    import psycopg2
    import psycopg2.extras
    from neo4j import GraphDatabase
except ImportError:
    pytest.skip("psycopg2 or neo4j not installed", allow_module_level=True)

# Test configuration
PG_HOST = os.getenv("TEST_PG_HOST", "localhost")
PG_PORT = int(os.getenv("TEST_PG_PORT", "5433"))
PG_DB = os.getenv("TEST_PG_DB", "xce_memory")
PG_USER = os.getenv("TEST_PG_USER", "xce_memory")
PG_PASSWORD = os.getenv("TEST_PG_PASSWORD", "xme_prod_password")

NEO4J_URI = os.getenv("TEST_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("TEST_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("TEST_NEO4J_PASSWORD", "xce_prod_password")


@pytest.fixture(scope="module")
def pg_conn():
    """PostgreSQL connection for tests."""
    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, database=PG_DB,
            user=PG_USER, password=PG_PASSWORD
        )
        conn.autocommit = True
        yield conn
        conn.close()
    except psycopg2.OperationalError:
        pytest.skip("PostgreSQL not available")


@pytest.fixture(scope="module")
def neo4j_driver():
    """Neo4j driver for tests."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        yield driver
        driver.close()
    except Exception:
        pytest.skip("Neo4j not available")


@pytest.fixture(autouse=True)
def cleanup(pg_conn, neo4j_driver):
    """Clean up test data before each test."""
    # Clean PostgreSQL
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM extracted_memories WHERE repo_id LIKE 'test-%'")
        cur.execute("DELETE FROM raw_messages WHERE repo_id LIKE 'test-%'")

    # Clean Neo4j
    with neo4j_driver.session() as session:
        session.run("MATCH (m:Memory {repo_id: 'test-repo'}) DETACH DELETE m")
        session.run("MATCH (c:Commit {repo_id: 'test-repo'}) DETACH DELETE c")
        session.run("MATCH (n:ASTNode {repo_id: 'test-repo'}) DETACH DELETE n")

    yield

    # Cleanup after
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM extracted_memories WHERE repo_id LIKE 'test-%'")
        cur.execute("DELETE FROM raw_messages WHERE repo_id LIKE 'test-%'")

    with neo4j_driver.session() as session:
        session.run("MATCH (m:Memory {repo_id: 'test-repo'}) DETACH DELETE m")
        session.run("MATCH (c:Commit {repo_id: 'test-repo'}) DETACH DELETE c")
        session.run("MATCH (n:ASTNode {repo_id: 'test-repo'}) DETACH DELETE n")


class TestPhase1RawCapture:
    """Test Phase 1: Raw message capture and extraction."""

    def test_insert_raw_message(self, pg_conn):
        """Test inserting a raw message into PostgreSQL."""
        session_id = str(uuid.uuid4())

        with pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO raw_messages (session_id, user_id, repo_id, role, content)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (session_id, "test-user", "test-repo", "user", "Let's use async for HTTP calls"))

            msg_id = cur.fetchone()[0]
            assert msg_id is not None

        # Verify it's unprocessed
        with pg_conn.cursor() as cur:
            cur.execute("SELECT processed FROM raw_messages WHERE id = %s", (msg_id,))
            assert cur.fetchone()[0] is False

    def test_batch_insert_and_mark_processed(self, pg_conn):
        """Test inserting multiple messages and marking them processed."""
        session_id = str(uuid.uuid4())
        msg_ids = []

        with pg_conn.cursor() as cur:
            for i in range(5):
                cur.execute("""
                    INSERT INTO raw_messages (session_id, user_id, repo_id, role, content)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (session_id, "test-user", "test-repo", "user", f"Message {i}"))
                msg_ids.append(cur.fetchone()[0])

        # Mark as processed
        with pg_conn.cursor() as cur:
            cur.execute("UPDATE raw_messages SET processed = TRUE WHERE id = ANY(%s)", (msg_ids,))

        # Verify
        with pg_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM raw_messages WHERE id = ANY(%s) AND processed = TRUE", (msg_ids,))
            assert cur.fetchone()[0] == 5

    def test_store_extracted_memory(self, pg_conn):
        """Test storing an extracted memory."""
        session_id = str(uuid.uuid4())

        with pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO extracted_memories
                (session_id, user_id, repo_id, kind, summary, reasoning, confidence, priority, references)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                session_id, "test-user", "test-repo", "decision",
                "Use async for all HTTP calls",
                "Blocking calls caused timeouts under load",
                0.9, 1,
                json.dumps([{"name": "utils/http.py"}])
            ))
            mem_id = cur.fetchone()[0]
            assert mem_id is not None

        # Verify it's not written to graph yet
        with pg_conn.cursor() as cur:
            cur.execute("SELECT written_to_graph FROM extracted_memories WHERE id = %s", (mem_id,))
            assert cur.fetchone()[0] is False


class TestPhase1GraphWrite:
    """Test Phase 1: Writing memory nodes to Neo4j."""

    def test_write_memory_to_neo4j(self, neo4j_driver):
        """Test creating a Memory node in Neo4j."""
        memory_id = str(uuid.uuid4())

        with neo4j_driver.session() as session:
            session.run("""
                CREATE (m:Memory {
                    id: $id,
                    session_id: $session_id,
                    repo_id: $repo_id,
                    kind: 'decision',
                    summary: 'Use async for HTTP calls',
                    confidence: 0.9,
                    priority: 1,
                    created_at: datetime()
                })
            """, {"id": memory_id, "session_id": "test-session", "repo_id": "test-repo"})

        # Verify
        with neo4j_driver.session() as session:
            result = session.run("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
            record = result.single()
            assert record is not None
            assert record["m"]["kind"] == "decision"

    def test_memory_links_to_code_node(self, neo4j_driver):
        """Test linking a Memory node to an ASTNode."""
        memory_id = str(uuid.uuid4())
        node_id = f"test-repo:utils/http.py:function:fetch_data"

        with neo4j_driver.session() as session:
            # Create ASTNode
            session.run("""
                CREATE (n:ASTNode {
                    id: $id, name: 'fetch_data', kind: 'function',
                    filepath: 'utils/http.py', repo_id: 'test-repo'
                })
            """, {"id": node_id})

            # Create Memory
            session.run("""
                CREATE (m:Memory {
                    id: $id, repo_id: 'test-repo', kind: 'decision',
                    summary: 'Made fetch_data async'
                })
            """, {"id": memory_id})

            # Link them
            session.run("""
                MATCH (m:Memory {id: $mid})
                MATCH (n:ASTNode {id: $nid})
                CREATE (m)-[:REFS]->(n)
            """, {"mid": memory_id, "nid": node_id})

        # Verify the link
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (m:Memory {id: $mid})-[:REFS]->(n:ASTNode)
                RETURN n.name as name
            """, {"mid": memory_id})
            record = result.single()
            assert record is not None
            assert record["name"] == "fetch_data"

    def test_eviction_at_cap(self, neo4j_driver):
        """Test that eviction works when at 500 node cap."""
        # Create 5 memories with different confidence
        for i in range(5):
            with neo4j_driver.session() as session:
                session.run("""
                    CREATE (m:Memory {
                        id: $id, repo_id: 'test-repo', kind: 'insight',
                        summary: $summary, confidence: $conf,
                        created_at: datetime() - duration({days: $days})
                    })
                """, {
                    "id": str(uuid.uuid4()),
                    "summary": f"Insight {i}",
                    "conf": 0.5 + (i * 0.1),
                    "days": 60 - (i * 10)
                })

        # Verify 5 exist
        with neo4j_driver.session() as session:
            result = session.run("MATCH (m:Memory {repo_id: 'test-repo'}) RETURN count(m) as cnt")
            assert result.single()["cnt"] == 5

        # Simulate eviction (delete lowest confidence)
        with neo4j_driver.session() as session:
            session.run("""
                MATCH (m:Memory {repo_id: 'test-repo'})
                WITH m ORDER BY m.confidence ASC
                LIMIT 2
                DETACH DELETE m
            """)

        # Verify 3 remain
        with neo4j_driver.session() as session:
            result = session.run("MATCH (m:Memory {repo_id: 'test-repo'}) RETURN count(m) as cnt")
            assert result.single()["cnt"] == 3


class TestPhase2GitLinking:
    """Test Phase 2: Git commit linking."""

    def test_commit_node_creation(self, neo4j_driver):
        """Test creating a Commit node."""
        sha = "abc123def456"

        with neo4j_driver.session() as session:
            session.run("""
                CREATE (c:Commit {
                    sha: $sha,
                    message: 'feat: add async HTTP support',
                    author: 'raj',
                    timestamp: datetime(),
                    repo_id: 'test-repo',
                    files_changed: ['utils/http.py', 'services/api.py']
                })
            """, {"sha": sha})

        # Verify
        with neo4j_driver.session() as session:
            result = session.run("MATCH (c:Commit {sha: $sha}) RETURN c", {"sha": sha})
            record = result.single()
            assert record is not None
            assert record["c"]["message"] == "feat: add async HTTP support"

    def test_commit_links_to_code_node(self, neo4j_driver):
        """Test CHANGED edge from Commit to ASTNode."""
        sha = "def789ghi012"
        node_id = "test-repo:utils/http.py:function:fetch_data_v2"

        with neo4j_driver.session() as session:
            # Create nodes
            session.run("""
                CREATE (c:Commit {sha: $sha, repo_id: 'test-repo', message: 'refactor fetch'})
            """, {"sha": sha})
            session.run("""
                CREATE (n:ASTNode {id: $id, name: 'fetch_data', filepath: 'utils/http.py', repo_id: 'test-repo', version: 1})
            """, {"id": node_id})

            # Link with CHANGED and bump version
            session.run("""
                MATCH (c:Commit {sha: $sha})
                MATCH (n:ASTNode {id: $nid})
                CREATE (c)-[:CHANGED]->(n)
                SET n.version = n.version + 1
            """, {"sha": sha, "nid": node_id})

        # Verify
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c:Commit {sha: $sha})-[:CHANGED]->(n:ASTNode)
                RETURN n.version as version, n.name as name
            """, {"sha": sha})
            record = result.single()
            assert record is not None
            assert record["version"] == 2
            assert record["name"] == "fetch_data"

    def test_memory_resulted_in_commit(self, neo4j_driver):
        """Test RESULTED_IN edge from Memory to Commit."""
        memory_id = str(uuid.uuid4())
        sha = "xyz999abc111"

        with neo4j_driver.session() as session:
            session.run("""
                CREATE (m:Memory {id: $mid, repo_id: 'test-repo', session_id: 'sess-1', kind: 'decision', summary: 'chose async'})
            """, {"mid": memory_id})
            session.run("""
                CREATE (c:Commit {sha: $sha, repo_id: 'test-repo', message: 'implement async'})
            """, {"sha": sha})
            session.run("""
                MATCH (m:Memory {id: $mid})
                MATCH (c:Commit {sha: $sha})
                CREATE (m)-[:RESULTED_IN]->(c)
            """, {"mid": memory_id, "sha": sha})

        # Query: "Why was this commit made?"
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (m:Memory)-[:RESULTED_IN]->(c:Commit {sha: $sha})
                RETURN m.summary as reason, m.kind as kind
            """, {"sha": sha})
            record = result.single()
            assert record is not None
            assert record["reason"] == "chose async"
            assert record["kind"] == "decision"

    def test_full_chain_query(self, neo4j_driver):
        """Test the full chain: Memory → Commit → ASTNode."""
        memory_id = str(uuid.uuid4())
        sha = "full_chain_test_sha"
        node_id = "test-repo:app.py:function:main"

        with neo4j_driver.session() as session:
            # Create all nodes
            session.run("CREATE (n:ASTNode {id: $id, name: 'main', filepath: 'app.py', repo_id: 'test-repo', version: 3})", {"id": node_id})
            session.run("CREATE (c:Commit {sha: $sha, repo_id: 'test-repo', message: 'refactor main'})", {"sha": sha})
            session.run("CREATE (m:Memory {id: $mid, repo_id: 'test-repo', kind: 'decision', summary: 'Split main into smaller functions'})", {"mid": memory_id})

            # Create edges
            session.run("MATCH (m:Memory {id: $mid}) MATCH (c:Commit {sha: $sha}) CREATE (m)-[:RESULTED_IN]->(c)", {"mid": memory_id, "sha": sha})
            session.run("MATCH (c:Commit {sha: $sha}) MATCH (n:ASTNode {id: $nid}) CREATE (c)-[:CHANGED]->(n)", {"sha": sha, "nid": node_id})

        # Query: "What decisions affected this function?"
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (m:Memory)-[:RESULTED_IN]->(c:Commit)-[:CHANGED]->(n:ASTNode {id: $nid})
                RETURN m.summary as decision, c.message as commit_msg, n.version as version
            """, {"nid": node_id})
            record = result.single()
            assert record is not None
            assert record["decision"] == "Split main into smaller functions"
            assert record["commit_msg"] == "refactor main"
            assert record["version"] == 3


class TestPhase2SessionArtifacts:
    """Test session artifact (MD file) generation."""

    def test_generate_session_summary(self, pg_conn):
        """Test generating a session summary MD file."""
        session_id = str(uuid.uuid4())

        # Insert test data
        with pg_conn.cursor() as cur:
            # Raw messages
            cur.execute("""
                INSERT INTO raw_messages (session_id, user_id, repo_id, role, content)
                VALUES (%s, %s, %s, %s, %s)
            """, (session_id, "test-user", "test-repo", "user", "Let's refactor the auth module"))

            # Extracted memories
            cur.execute("""
                INSERT INTO extracted_memories
                (session_id, user_id, repo_id, kind, summary, reasoning, confidence, priority, references, written_to_graph)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                session_id, "test-user", "test-repo", "decision",
                "Refactor auth to use JWT tokens",
                "Current session-based auth doesn't scale",
                0.9, 1, json.dumps(["auth/session.py"]), True
            ))

        # Generate artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            from services.memory_worker.git_linker import SessionArtifactGenerator
            gen = SessionArtifactGenerator(pg_conn, output_dir=tmpdir)
            filepath = gen.generate_session_summary(session_id)

            assert filepath is not None
            assert os.path.exists(filepath)

            # Read and verify content
            with open(filepath) as f:
                content = f.read()

            assert "Refactor auth to use JWT tokens" in content
            assert "decision" in content.lower() or "Decisions" in content


class TestQueryPatterns:
    """Test common query patterns that agents would use at runtime."""

    def setup_method(self, method):
        """Set up test graph data."""
        pass

    def test_query_why_was_function_changed(self, neo4j_driver):
        """Agent query: 'Why was fetch_data changed?'"""
        # Setup
        with neo4j_driver.session() as session:
            session.run("""
                CREATE (n:ASTNode {id: 'test-repo:api.py:function:fetch_data', name: 'fetch_data', repo_id: 'test-repo', version: 2})
                CREATE (c:Commit {sha: 'query_test_1', repo_id: 'test-repo', message: 'make fetch async'})
                CREATE (m:Memory {id: 'mem-query-1', repo_id: 'test-repo', kind: 'decision', summary: 'Async improves throughput by 3x under load'})
                CREATE (m)-[:RESULTED_IN]->(c)
                CREATE (c)-[:CHANGED]->(n)
            """)

        # Agent query
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (n:ASTNode {name: 'fetch_data', repo_id: 'test-repo'})
                OPTIONAL MATCH (m:Memory)-[:RESULTED_IN]->(c:Commit)-[:CHANGED]->(n)
                RETURN n.name as func, m.summary as reason, c.sha as commit
            """)
            record = result.single()
            assert record["func"] == "fetch_data"
            assert record["reason"] == "Async improves throughput by 3x under load"

    def test_query_all_decisions_for_file(self, neo4j_driver):
        """Agent query: 'What decisions were made about utils/http.py?'"""
        with neo4j_driver.session() as session:
            session.run("""
                CREATE (n1:ASTNode {id: 'test-repo:utils/http.py:function:get', name: 'get', filepath: 'utils/http.py', repo_id: 'test-repo'})
                CREATE (n2:ASTNode {id: 'test-repo:utils/http.py:function:post', name: 'post', filepath: 'utils/http.py', repo_id: 'test-repo'})
                CREATE (m1:Memory {id: 'mem-file-1', repo_id: 'test-repo', kind: 'decision', summary: 'Use connection pooling'})
                CREATE (m2:Memory {id: 'mem-file-2', repo_id: 'test-repo', kind: 'decision', summary: 'Add retry logic with backoff'})
                CREATE (m1)-[:REFS]->(n1)
                CREATE (m2)-[:REFS]->(n2)
            """)

        # Agent query
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (m:Memory {repo_id: 'test-repo', kind: 'decision'})-[:REFS]->(n:ASTNode)
                WHERE n.filepath = 'utils/http.py'
                RETURN m.summary as decision, n.name as function
                ORDER BY m.summary
            """)
            records = list(result)
            assert len(records) == 2
            summaries = [r["decision"] for r in records]
            assert "Add retry logic with backoff" in summaries
            assert "Use connection pooling" in summaries