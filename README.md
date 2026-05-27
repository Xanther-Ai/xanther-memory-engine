# Xanther Memory Engine (XME)

**Persistent memory for coding agents — linked to your codebase architecture.**

[Website](https://xanther.ai) • [Dashboard](https://app.xanther.ai) • [Discord](https://discord.com/invite/p27qtGkTYw) • [XCE MCP Server](https://github.com/Xanther-Ai/xce-mcp) • [Blog](https://medium.com/@xanther.ai)

---

## What is XME?

Xanther Memory Engine gives your coding agent **persistent memory** across sessions. It remembers decisions, preferences, bugs found, and architectural context — linked directly to your codebase graph via XCE.

**The problem:** Every new chat, your agent starts from zero. It doesn't remember what it tried, what decisions were made, or your preferences.

**The solution:** XME captures session knowledge, extracts structured memories via LLM, and stores them in a graph linked to your code architecture.

## How It Works

```
Agent Session → Raw Messages (PostgreSQL/TimescaleDB)
                    ↓ (every 5 min)
              LLM Extraction
                    ↓
         Distilled Memory Nodes (Neo4j)
                    ↓
         Linked to XCE Architecture Graph
```

1. **Capture** — Raw messages from coding sessions stored in TimescaleDB
2. **Extract** — LLM extracts decisions, bugs, insights from raw messages
3. **Store** — Distilled memory nodes written to Neo4j (max 500/repo)
4. **Link** — Memories linked to code architecture nodes via XCE graph
5. **Recall** — Agent queries memories by topic, time, or relevance

## Components

| Component | Path | Purpose |
|-----------|------|---------|
| Worker | `services/worker/` | Main processing loop — extracts and stores memories |
| Pruner | `services/pruner/` | Evicts old/low-confidence memories at cap |
| SDK | `packages/sdk/` | TypeScript SDK for MCP tool integration |
| Tests | `tests/` | Integration tests |

## Memory Types

- **Decisions** — "We chose X because Y"
- **Bugs** — "Found race condition in auth/middleware.py"
- **Insights** — "This module uses the Observer pattern"
- **Questions** — "How does the cache invalidation work?" (with answer)

## Storage

- **PostgreSQL (TimescaleDB)** — Raw messages + extracted memories
- **Neo4j** — Distilled memory nodes linked to XCE architecture graph
- **Cap: 500 nodes/repo** — Evicts lowest confidence + oldest when full

## Setup

```bash
# Environment
export PG_HOST=localhost PG_PORT=5432 PG_DB=xce_memory
export NEO4J_URI=bolt://localhost:7687
export LLM_API_KEY=your_openrouter_key

# Run
cd services/worker
pip install -r requirements.txt
python handler.py
```

## MCP Tools (Coming Soon)

| Tool | Description |
|------|-------------|
| `xme_remember` | Store a memory node |
| `xme_recall` | Query memories by topic/time/relevance |
| `xme_session_state` | Get/set session state |
| `xme_preferences` | Read/write user preferences |
| `xme_history` | Query decision history |

## Relationship to XCE

- **XCE** = understands your code (architecture, symbols, dependencies)
- **XME** = remembers your sessions (decisions, preferences, history)
- **Together** = agent knows the code AND remembers what happened

## License

MIT
