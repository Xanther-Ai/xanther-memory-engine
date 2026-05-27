"""
Git Linker — Phase 2 of XME

Links git commits to sessions and tracks code changes with version history.
Runs as part of the XME worker after extraction.

Flow:
1. Read recent commits from git
2. Match commits to active sessions by timestamp
3. Detect changed AST nodes (functions/classes that were modified)
4. Create Commit nodes in Neo4j
5. Link: Memory → RESULTED_IN → Commit → CHANGED → ASTNode
6. Bump version on changed ASTNodes
"""

import os
import json
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Optional
from dataclasses import dataclass

import psycopg2
import psycopg2.extras
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


@dataclass
class GitCommit:
    sha: str
    message: str
    author: str
    timestamp: datetime
    files_changed: list  # list of file paths


class GitClient:
    """Reads git history from a local repo clone."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def get_recent_commits(self, since_hours: int = 6) -> list:
        """Get commits from the last N hours."""
        since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        since_str = since.strftime("%Y-%m-%dT%H:%M:%S")

        try:
            # Get commit log with files
            result = subprocess.run(
                ["git", "log", f"--since={since_str}", "--format=%H|%s|%an|%aI", "--name-only"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"git log failed: {result.stderr}")
                return []

            commits = []
            current_commit = None

            for line in result.stdout.strip().split("\n"):
                if not line:
                    if current_commit:
                        commits.append(current_commit)
                        current_commit = None
                    continue

                if "|" in line and line.count("|") >= 3:
                    # This is a commit header line
                    if current_commit:
                        commits.append(current_commit)

                    parts = line.split("|", 3)
                    current_commit = GitCommit(
                        sha=parts[0],
                        message=parts[1],
                        author=parts[2],
                        timestamp=datetime.fromisoformat(parts[3]),
                        files_changed=[]
                    )
                elif current_commit:
                    # This is a file path
                    current_commit.files_changed.append(line.strip())

            if current_commit:
                commits.append(current_commit)

            return commits

        except subprocess.TimeoutExpired:
            logger.error("git log timed out")
            return []
        except Exception as e:
            logger.error(f"Failed to get git commits: {e}")
            return []

    def get_diff_for_file(self, sha: str, filepath: str) -> str:
        """Get the diff for a specific file in a commit."""
        try:
            result = subprocess.run(
                ["git", "diff", f"{sha}~1..{sha}", "--", filepath],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout[:2000]  # Truncate large diffs
        except Exception:
            return ""

    def get_changed_functions(self, sha: str, filepath: str) -> list:
        """Get function/class names that changed in a commit for a file."""
        try:
            # Use git diff with function context
            result = subprocess.run(
                ["git", "diff", f"{sha}~1..{sha}", "-U0", "--function-context", "--", filepath],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse @@ lines for function names
            functions = []
            for line in result.stdout.split("\n"):
                if line.startswith("@@") and "@@" in line[2:]:
                    # Extract function context from @@ -x,y +a,b @@ function_name
                    parts = line.split("@@")
                    if len(parts) >= 3:
                        func_context = parts[2].strip()
                        if func_context:
                            functions.append(func_context.split("(")[0].strip())

            return list(set(functions))
        except Exception:
            return []


class GitLinker:
    """Links git commits to sessions and code nodes in Neo4j."""

    def __init__(self, neo4j_driver, pg_conn, repo_path: str, repo_id: str):
        self.driver = neo4j_driver
        self.pg_conn = pg_conn
        self.git = GitClient(repo_path)
        self.repo_id = repo_id

    def find_session_for_commit(self, commit: GitCommit) -> Optional[str]:
        """Find the session that was active when this commit was made."""
        # Look for sessions within 30 minutes of the commit
        window_start = commit.timestamp - timedelta(minutes=30)
        window_end = commit.timestamp + timedelta(minutes=5)

        with self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT session_id
                FROM raw_messages
                WHERE repo_id = %s
                AND created_at BETWEEN %s AND %s
                ORDER BY session_id
                LIMIT 1
            """, (self.repo_id, window_start, window_end))

            row = cur.fetchone()
            return row["session_id"] if row else None

    def create_commit_node(self, commit: GitCommit) -> bool:
        """Create a Commit node in Neo4j."""
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (c:Commit {sha: $sha})
                    SET c.message = $message,
                        c.author = $author,
                        c.timestamp = datetime($timestamp),
                        c.repo_id = $repo_id,
                        c.files_changed = $files
                """, {
                    "sha": commit.sha,
                    "message": commit.message,
                    "author": commit.author,
                    "timestamp": commit.timestamp.isoformat(),
                    "repo_id": self.repo_id,
                    "files": commit.files_changed
                })
            return True
        except Exception as e:
            logger.error(f"Failed to create commit node {commit.sha}: {e}")
            return False

    def link_commit_to_session(self, commit: GitCommit, session_id: str) -> bool:
        """Link commit to the session's memory nodes."""
        try:
            with self.driver.session() as session:
                # Find memory nodes from this session and link them to the commit
                session.run("""
                    MATCH (m:Memory {session_id: $session_id, repo_id: $repo_id})
                    MATCH (c:Commit {sha: $sha})
                    MERGE (m)-[:RESULTED_IN]->(c)
                """, {
                    "session_id": session_id,
                    "sha": commit.sha,
                    "repo_id": self.repo_id
                })
            return True
        except Exception as e:
            logger.error(f"Failed to link commit to session: {e}")
            return False

    def link_commit_to_code_nodes(self, commit: GitCommit) -> int:
        """Link commit to changed ASTNodes and bump their version."""
        linked = 0

        try:
            with self.driver.session() as session:
                for filepath in commit.files_changed:
                    # Find ASTNodes in this file
                    result = session.run("""
                        MATCH (n:ASTNode)
                        WHERE n.filepath = $filepath AND n.repo_id = $repo_id
                        RETURN n.id as node_id, n.name as name
                    """, {"filepath": filepath, "repo_id": self.repo_id})

                    nodes = [r for r in result]

                    for node in nodes:
                        # Create CHANGED edge and bump version
                        session.run("""
                            MATCH (c:Commit {sha: $sha})
                            MATCH (n:ASTNode {id: $node_id})
                            MERGE (c)-[:CHANGED]->(n)
                            SET n.version = COALESCE(n.version, 0) + 1,
                                n.last_changed_at = datetime($timestamp),
                                n.last_commit_sha = $sha
                        """, {
                            "sha": commit.sha,
                            "node_id": node["node_id"],
                            "timestamp": commit.timestamp.isoformat()
                        })
                        linked += 1

        except Exception as e:
            logger.error(f"Failed to link commit to code nodes: {e}")

        return linked

    def process_recent_commits(self, since_hours: int = 6) -> dict:
        """Main entry point: process all recent commits."""
        commits = self.git.get_recent_commits(since_hours)

        if not commits:
            return {"commits": 0, "linked_sessions": 0, "linked_nodes": 0}

        stats = {"commits": len(commits), "linked_sessions": 0, "linked_nodes": 0}

        for commit in commits:
            # 1. Create commit node
            if not self.create_commit_node(commit):
                continue

            # 2. Find and link to session
            session_id = self.find_session_for_commit(commit)
            if session_id:
                self.link_commit_to_session(commit, session_id)
                stats["linked_sessions"] += 1

            # 3. Link to changed code nodes
            linked = self.link_commit_to_code_nodes(commit)
            stats["linked_nodes"] += linked

        logger.info(f"Git linking complete: {stats}")
        return stats


class SessionArtifactGenerator:
    """Generates session summary MD files after sessions complete."""

    def __init__(self, pg_conn, output_dir: str = "memory/sessions"):
        self.pg_conn = pg_conn
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_session_summary(self, session_id: str) -> Optional[str]:
        """Generate a markdown summary for a completed session."""
        with self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get extracted memories for this session
            cur.execute("""
                SELECT kind, summary, reasoning, confidence, references, created_at
                FROM extracted_memories
                WHERE session_id = %s
                ORDER BY created_at ASC
            """, (session_id,))
            memories = cur.fetchall()

            if not memories:
                return None

            # Get session metadata
            cur.execute("""
                SELECT repo_id, user_id, MIN(created_at) as started, MAX(created_at) as ended
                FROM raw_messages
                WHERE session_id = %s
                GROUP BY repo_id, user_id
            """, (session_id,))
            meta = cur.fetchone()

            if not meta:
                return None

        # Generate MD content
        md = f"""# Session Summary: {session_id[:8]}
**Date:** {meta['started'].strftime('%Y-%m-%d %H:%M')} — {meta['ended'].strftime('%H:%M')}
**Repo:** {meta['repo_id']}
**User:** {meta['user_id']}

"""
        # Group by kind
        decisions = [m for m in memories if m['kind'] == 'decision']
        bugs = [m for m in memories if m['kind'] == 'bug']
        insights = [m for m in memories if m['kind'] == 'insight']
        questions = [m for m in memories if m['kind'] == 'question']

        if decisions:
            md += "## Decisions\n"
            for d in decisions:
                md += f"- **{d['summary']}**\n"
                if d['reasoning']:
                    md += f"  - Reasoning: {d['reasoning']}\n"
                if d['references']:
                    refs = json.loads(d['references']) if isinstance(d['references'], str) else d['references']
                    md += f"  - Files: {', '.join(str(r) for r in refs[:5])}\n"
            md += "\n"

        if bugs:
            md += "## Bugs Discovered\n"
            for b in bugs:
                md += f"- {b['summary']}\n"
            md += "\n"

        if insights:
            md += "## Insights\n"
            for i in insights:
                md += f"- {i['summary']}\n"
            md += "\n"

        if questions:
            md += "## Questions & Answers\n"
            for q in questions:
                md += f"- {q['summary']}\n"
            md += "\n"

        # Write to file
        filename = f"session-{meta['started'].strftime('%Y-%m-%d')}-{session_id[:8]}.md"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            f.write(md)

        logger.info(f"Generated session summary: {filepath}")
        return filepath