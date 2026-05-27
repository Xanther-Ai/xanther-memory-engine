"""
XME MCP Server (Xanther Memory Engine)

FastAPI-based MCP SSE server exposing memory tools:
- xme_remember: Store a memory directly
- xme_recall: Query memories by topic, time, kind, repo
- xme_session_state: Get/set session state
- xme_preferences: Read/write user preferences
- xme_history: Query decision history

Protocol: MCP SSE (SSE at /sse, messages at /messages)
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Any
from contextlib import asynccontextmanager

import asyncpg
from neo4j import AsyncGraphDatabase
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [XME-MCP] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# --- Configuration ---

class Config:
    PG_HOST = os.getenv("PG_HOST", "xme-postgres")
    PG_PORT = int(os.getenv("PG_PORT", "5432"))
    PG_DB = os.getenv("PG_DB", "xce_memory")
    PG_USER = os.getenv("PG_USER", "xce_memory")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "xme_prod_password")

    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://xce-neo4j:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "xce_prod_password")

    SERVER_HOST = os.getenv("MCP_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("MCP_PORT", "8100"))


config = Config()


# --- Database Connections ---

pg_pool: Optional[asyncpg.Pool] = None
neo4j_driver = None


async def init_pg():
    global pg_pool
    pg_pool = await asyncpg.create_pool(
        host=config.PG_HOST,
        port=config.PG_PORT,
        database=config.PG_DB,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
        min_size=2,
        max_size=10,
    )
    logger.info(f"PostgreSQL pool created ({config.PG_HOST}:{config.PG_PORT})")


async def init_neo4j():
    global neo4j_driver
    neo4j_driver = AsyncGraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )
    logger.info(f"Neo4j driver created ({config.NEO4J_URI})")


async def close_connections():
    global pg_pool, neo4j_driver
    if pg_pool:
        await pg_pool.close()
    if neo4j_driver:
        await neo4j_driver.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pg()
    await init_neo4j()
    logger.info("XME MCP Server started")
    yield
    await close_connections()
    logger.info("XME MCP Server stopped")


# --- FastAPI App ---

app = FastAPI(title="XME MCP Server", lifespan=lifespan)


# --- MCP Tool Definitions ---

TOOLS = [
    {
        "name": "xme_remember",
        "description": "Store a memory directly into the Xanther Memory Engine. Use this to persist decisions, insights, bugs, patterns, or any knowledge worth remembering across sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["decision", "bug", "insight", "pattern", "question", "preference", "state"],
                    "description": "The type of memory to store",
                },
                "summary": {
                    "type": "string",
                    "description": "A concise summary of the memory (1-2 sentences)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why this memory matters or additional context",
                },
                "references": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths, function names, or other code references",
                },
                "repo_id": {
                    "type": "string",
                    "description": "Repository identifier",
                },
                "session_id": {
                    "type": "string",
                    "description": "Current session identifier",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                },
            },
            "required": ["kind", "summary", "repo_id", "session_id", "user_id"],
        },
    },
    {
        "name": "xme_recall",
        "description": "Query memories from the Xanther Memory Engine. Search by topic (semantic match on summary), filter by kind, repo, and time range.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to search memories by topic",
                },
                "kind": {
                    "type": "string",
                    "enum": ["decision", "bug", "insight", "pattern", "question", "preference", "state"],
                    "description": "Filter by memory kind",
                },
                "repo_id": {
                    "type": "string",
                    "description": "Filter by repository",
                },
                "user_id": {
                    "type": "string",
                    "description": "Filter by user",
                },
                "since": {
                    "type": "string",
                    "description": "ISO 8601 timestamp — only return memories after this time",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 20)",
                },
            },
            "required": ["query", "user_id"],
        },
    },
    {
        "name": "xme_session_state",
        "description": "Get or set session state. Use 'get' to retrieve current state, 'set' to store state for the session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set"],
                    "description": "Whether to get or set session state",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session identifier",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                },
                "key": {
                    "type": "string",
                    "description": "State key to get/set",
                },
                "value": {
                    "description": "Value to set (required for 'set' action)",
                },
            },
            "required": ["action", "session_id", "user_id", "key"],
        },
    },
    {
        "name": "xme_preferences",
        "description": "Read or write user preferences. Preferences persist across sessions and repos.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "list"],
                    "description": "Whether to get, set, or list preferences",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                },
                "key": {
                    "type": "string",
                    "description": "Preference key (required for get/set)",
                },
                "value": {
                    "description": "Preference value (required for set)",
                },
                "repo_id": {
                    "type": "string",
                    "description": "Optional repo scope for repo-specific preferences",
                },
            },
            "required": ["action", "user_id"],
        },
    },
    {
        "name": "xme_history",
        "description": "Query decision history. Returns decisions and their reasoning in chronological order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                },
                "repo_id": {
                    "type": "string",
                    "description": "Filter by repository",
                },
                "since": {
                    "type": "string",
                    "description": "ISO 8601 timestamp — only return decisions after this time",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 20)",
                },
            },
            "required": ["user_id"],
        },
    },
]


# --- Tool Handlers ---

async def handle_xme_remember(params: dict) -> dict:
    """Store a memory directly."""
    memory_id = str(uuid.uuid4())
    async with pg_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO extracted_memories
                (id, session_id, user_id, repo_id, kind, summary, reasoning, confidence, priority, refs, source_message_ids)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            uuid.UUID(memory_id),
            params["session_id"],
            params["user_id"],
            params["repo_id"],
            params["kind"],
            params["summary"],
            params.get("reasoning"),
            0.95,  # Direct memories get high confidence
            _infer_priority(params["kind"]),
            json.dumps(params.get("references", [])),
            [],
        )
    return {
        "stored": True,
        "memory_id": memory_id,
        "kind": params["kind"],
        "summary": params["summary"],
    }


async def handle_xme_recall(params: dict) -> dict:
    """Query memories by topic with filters."""
    query = params["query"]
    user_id = params["user_id"]
    kind = params.get("kind")
    repo_id = params.get("repo_id")
    since = params.get("since")
    limit = min(params.get("limit", 20), 100)

    # Build query with filters
    conditions = ["user_id = $1"]
    args: list[Any] = [user_id]
    idx = 2

    if kind:
        conditions.append(f"kind = ${idx}")
        args.append(kind)
        idx += 1

    if repo_id:
        conditions.append(f"repo_id = ${idx}")
        args.append(repo_id)
        idx += 1

    if since:
        conditions.append(f"created_at >= ${idx}::timestamptz")
        args.append(since)
        idx += 1

    # Semantic search: use ILIKE for now (upgrade to pgvector later)
    # Split query into keywords for basic relevance matching
    keywords = [w.strip() for w in query.lower().split() if len(w.strip()) > 2]
    if keywords:
        keyword_conditions = []
        for kw in keywords[:5]:  # Max 5 keywords
            keyword_conditions.append(f"LOWER(summary) LIKE ${idx}")
            args.append(f"%{kw}%")
            idx += 1
        conditions.append(f"({' OR '.join(keyword_conditions)})")

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT id, session_id, repo_id, kind, summary, reasoning, confidence, priority, refs, created_at
        FROM extracted_memories
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${idx}
    """
    args.append(limit)

    async with pg_pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)

    memories = []
    for row in rows:
        memories.append({
            "id": str(row["id"]),
            "session_id": row["session_id"],
            "repo_id": row["repo_id"],
            "kind": row["kind"],
            "summary": row["summary"],
            "reasoning": row["reasoning"],
            "confidence": row["confidence"],
            "priority": row["priority"],
            "refs": json.loads(row["refs"]) if row["refs"] else [],
            "created_at": row["created_at"].isoformat(),
        })

    return {"memories": memories, "count": len(memories), "query": query}


async def handle_xme_session_state(params: dict) -> dict:
    """Get or set session state."""
    action = params["action"]
    session_id = params["session_id"]
    user_id = params["user_id"]
    key = params["key"]

    async with pg_pool.acquire() as conn:
        if action == "get":
            row = await conn.fetchrow(
                "SELECT value, updated_at FROM session_state WHERE session_id = $1 AND key = $2",
                session_id,
                key,
            )
            if row:
                return {"key": key, "value": json.loads(row["value"]), "updated_at": row["updated_at"].isoformat()}
            return {"key": key, "value": None}

        elif action == "set":
            value = params.get("value")
            if value is None:
                return {"error": "value is required for set action"}

            await conn.execute(
                """
                INSERT INTO session_state (session_id, user_id, key, value, updated_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (session_id, key) DO UPDATE SET value = $4, updated_at = NOW()
                """,
                session_id,
                user_id,
                key,
                json.dumps(value),
            )
            return {"key": key, "stored": True}

    return {"error": f"Unknown action: {action}"}


async def handle_xme_preferences(params: dict) -> dict:
    """Read/write user preferences (stored as kind='preference' in extracted_memories)."""
    action = params["action"]
    user_id = params["user_id"]
    repo_id = params.get("repo_id", "__global__")

    async with pg_pool.acquire() as conn:
        if action == "get":
            key = params.get("key")
            if not key:
                return {"error": "key is required for get action"}

            row = await conn.fetchrow(
                """
                SELECT summary, reasoning, created_at FROM extracted_memories
                WHERE user_id = $1 AND repo_id = $2 AND kind = 'preference' AND summary LIKE $3
                ORDER BY created_at DESC LIMIT 1
                """,
                user_id,
                repo_id,
                f"[{key}]%",
            )
            if row:
                # Parse value from summary format: [key] value
                raw = row["summary"]
                value = raw.split("] ", 1)[1] if "] " in raw else raw
                return {"key": key, "value": value, "updated_at": row["created_at"].isoformat()}
            return {"key": key, "value": None}

        elif action == "set":
            key = params.get("key")
            value = params.get("value")
            if not key or value is None:
                return {"error": "key and value are required for set action"}

            # Store as a preference memory
            summary = f"[{key}] {json.dumps(value) if not isinstance(value, str) else value}"
            await conn.execute(
                """
                INSERT INTO extracted_memories
                    (session_id, user_id, repo_id, kind, summary, reasoning, confidence, priority, refs, source_message_ids)
                VALUES ($1, $2, $3, 'preference', $4, $5, 1.0, 0, '[]', '{}')
                """,
                f"pref-{user_id}",
                user_id,
                repo_id,
                summary,
                f"User preference: {key}",
            )
            return {"key": key, "stored": True}

        elif action == "list":
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (summary) summary, created_at
                FROM extracted_memories
                WHERE user_id = $1 AND kind = 'preference' AND (repo_id = $2 OR repo_id = '__global__')
                ORDER BY summary, created_at DESC
                """,
                user_id,
                repo_id,
            )
            prefs = {}
            for row in rows:
                raw = row["summary"]
                if "] " in raw and raw.startswith("["):
                    k = raw.split("]")[0][1:]
                    v = raw.split("] ", 1)[1]
                    prefs[k] = v
            return {"preferences": prefs, "count": len(prefs)}

    return {"error": f"Unknown action: {action}"}


async def handle_xme_history(params: dict) -> dict:
    """Query decision history."""
    user_id = params["user_id"]
    repo_id = params.get("repo_id")
    since = params.get("since")
    limit = min(params.get("limit", 20), 100)

    conditions = ["user_id = $1", "kind = 'decision'"]
    args: list[Any] = [user_id]
    idx = 2

    if repo_id:
        conditions.append(f"repo_id = ${idx}")
        args.append(repo_id)
        idx += 1

    if since:
        conditions.append(f"created_at >= ${idx}::timestamptz")
        args.append(since)
        idx += 1

    where_clause = " AND ".join(conditions)
    sql = f"""
        SELECT id, session_id, repo_id, summary, reasoning, confidence, refs, created_at
        FROM extracted_memories
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${idx}
    """
    args.append(limit)

    async with pg_pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)

    decisions = []
    for row in rows:
        decisions.append({
            "id": str(row["id"]),
            "session_id": row["session_id"],
            "repo_id": row["repo_id"],
            "summary": row["summary"],
            "reasoning": row["reasoning"],
            "confidence": row["confidence"],
            "refs": json.loads(row["refs"]) if row["refs"] else [],
            "created_at": row["created_at"].isoformat(),
        })

    return {"decisions": decisions, "count": len(decisions)}


def _infer_priority(kind: str) -> int:
    return {"bug": 0, "decision": 1, "question": 1, "insight": 2, "pattern": 3, "preference": 0, "state": 3}.get(kind, 2)


# --- Tool Dispatch ---

TOOL_HANDLERS = {
    "xme_remember": handle_xme_remember,
    "xme_recall": handle_xme_recall,
    "xme_session_state": handle_xme_session_state,
    "xme_preferences": handle_xme_preferences,
    "xme_history": handle_xme_history,
}


# --- MCP JSON-RPC Handling ---

async def handle_jsonrpc(request_body: dict) -> dict:
    """Handle a JSON-RPC 2.0 request per MCP protocol."""
    method = request_body.get("method")
    params = request_body.get("params", {})
    req_id = request_body.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "xme-mcp-server", "version": "0.1.0"},
            },
        }

    elif method == "notifications/initialized":
        # Client acknowledgment, no response needed
        return None

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {tool_name}"})}],
                    "isError": True,
                },
            }

        try:
            result = await handler(tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                    "isError": False,
                },
            }
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                    "isError": True,
                },
            }

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


# --- SSE Transport ---

# Active SSE sessions: session_id -> asyncio.Queue
sse_sessions: dict[str, asyncio.Queue] = {}


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE endpoint — client connects here to receive server messages."""
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    sse_sessions[session_id] = queue

    # Send the endpoint URL as the first message
    endpoint_url = f"/messages?session_id={session_id}"

    async def event_generator():
        # First event: tell client where to POST messages
        yield {"event": "endpoint", "data": endpoint_url}

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {"event": "message", "data": json.dumps(message)}
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}
        finally:
            sse_sessions.pop(session_id, None)

    return EventSourceResponse(event_generator())


@app.post("/messages")
async def messages_endpoint(request: Request):
    """Messages endpoint — client POSTs JSON-RPC requests here."""
    session_id = request.query_params.get("session_id")
    if not session_id or session_id not in sse_sessions:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid or missing session_id. Connect to /sse first."},
        )

    body = await request.json()
    response = await handle_jsonrpc(body)

    if response is not None:
        queue = sse_sessions.get(session_id)
        if queue:
            await queue.put(response)

    return Response(status_code=202)


@app.get("/health")
async def health():
    """Health check endpoint."""
    pg_ok = pg_pool is not None and not pg_pool._closed
    neo4j_ok = neo4j_driver is not None
    return {
        "status": "healthy" if (pg_ok and neo4j_ok) else "degraded",
        "postgres": "connected" if pg_ok else "disconnected",
        "neo4j": "connected" if neo4j_ok else "disconnected",
    }


# --- Entrypoint ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_level="info",
    )
