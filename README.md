<div align="center">

# Xanther Memory Engine (XME)

**The AI coding assistant memory layer that actually works.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Tests](https://github.com/Xanther-Ai/xanther-context-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Xanther-Ai/xanther-context-engine/actions)

</div>

---

> Your AI assistant re-reads your entire codebase every session. It forgets every decision you've made. It repeats the same failed approaches. Xanther fixes this.

XME gives AI coding assistants **persistent memory** across sessions — decisions remembered, failures not repeated, context always current. Works with Claude Code, Kiro, Cursor, Codex, and any MCP-compatible tool. No cloud required.

```bash
pip install xanther-memory-engine
xme hook install .          # 30 seconds to set up
xme start my-project        # memory starts now
```

---

## What Xanther does

Graphify maps what your code *is*. Xanther remembers what your team *did*.

Every session, Xanther captures what was discussed, extracts decisions and lessons learned, and makes them available at the start of the next session — without you having to re-explain anything.

```
Session 1:  "We decided to use FastAPI. Redis lock failed — timeout under load."
Session 2:  Agent already knows. Doesn't suggest Redis. Doesn't re-explain FastAPI.
Session 10: Full institutional memory. New team members onboard in minutes.
```

---

## Three memory layers

```
┌─────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│  EPISODIC        │   │  FACTS (Graph)    │   │  WORKING CONTEXT      │
│                  │   │                   │   │                       │
│  Full session    │   │  Decisions        │   │  Current task         │
│  transcripts     │   │  Attempts         │   │  Recent decisions     │
│  verbatim        │   │  Preferences      │   │  Next steps           │
│                  │   │  Conventions      │   │  Open questions       │
│  OpenSearch      │   │  Entities         │   │                       │
│  + SQLite FTS5   │   │  Neo4j            │   │  SQLite UPSERT        │
│  (fallback)      │   │  + vector dedup   │   │  per (project, user)  │
└─────────────────┘   └──────────────────┘   └──────────────────────┘
```

**Episodic** — raw session transcripts, full-text searchable. "What did we try last Tuesday?"

**Facts** — structured knowledge extracted from sessions. Decisions, failed approaches, preferences, conventions. Vector-deduplicated so the same fact never gets stored twice. Graph-linked so related facts surface together.

**Working Context** — the current state of a `(project, user)` pair. Always up-to-date via UPSERT. Injected into agent context at session start.

---

## Quickstart

**Install:**
```bash
pip install xanther-memory-engine
```

**Install hooks** (Kiro + Claude Code):
```bash
xme hook install .
```

That's it. XME now auto-captures every session. No other setup needed.

**Start a session manually:**
```bash
xme start my-project
```

**Search your memory:**
```bash
xme search my-project "why did we choose FastAPI"
xme facts my-project --type decision
```

**Launch the dashboard:**
```bash
xme dashboard
# Open http://localhost:8001
```

---

## How it works

**During a session**, two hooks fire silently:
- `promptSubmit` — buffers your message to `.xanther/turns/`
- `agentStop` — drains the buffer: extracts facts, updates context, saves episode

**At the start of the next session**, the agent gets a context block like:

```markdown
**Current task**: Refactor auth module
**Last session**: Moved JWT logic to dedicated auth service — success
**Recent decisions**:
- [VALIDATED] Use FastAPI for auth service — async support required
- [VALIDATED] PostgreSQL for main DB — ACID compliance
**Known failed approaches**:
- Redis distributed lock — timeout under load >1000 req/s
**Next steps**: Deploy auth service to staging
```

The agent starts informed. No re-explaining. No repeated mistakes.

---

## MCP tools

Xanther exposes 11 memory tools via MCP, working alongside 5 code intelligence tools:

| Tool | What it does |
|------|-------------|
| `xme_session_start` | Start session, get primed context for prompt injection |
| `xme_session_end` | End session: persist episode, extract facts, update context |
| `xme_add` | Add content to memory — Mem0-style UPSERT with deduplication |
| `xme_search` | Search across all 3 layers simultaneously |
| `xme_get_context` | Get current working context for a project+user |
| `xme_facts` | Query the fact graph — decisions, attempts, preferences |
| `xme_episodes` | Search past sessions |
| `xme_remember` | Explicitly store a typed fact |
| `xme_forget` | Soft-delete a memory node |
| `xme_export` | Export to Obsidian vault, wiki, or Graphify-compatible JSON |
| `xme_context_update` | Partial UPSERT of working context fields |

Add to your MCP config:
```json
{
  "mcpServers": {
    "xanther": {
      "command": "xce-mcp-server",
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_PASSWORD": "your-password"
      }
    }
  }
}
```

---

## Deduplication

Facts are stored once, not repeated. When you add new content, Xanther:

1. Embeds it using `all-MiniLM-L6-v2` (local, no API key)
2. Searches for similar existing facts (cosine similarity)
3. If similarity > 0.85: **merges** the new info into the existing fact
4. If no match: creates a new fact node

```
Session 3: "we use FastAPI"        → merges with existing FastAPI decision (sim=0.93)
Session 7: "decided on FastAPI"    → merges again (sim=0.91)
Session 15: "PostgreSQL migration" → new fact (sim=0.12 vs FastAPI)
```

Your fact graph stays clean even across dozens of sessions.

---

## Multi-user, multi-project

Every memory operation is scoped to `(project_id, user_id)`:

- **Team scope** — decisions, attempts, conventions are shared across users in a project
- **Personal scope** — sessions, preferences are per-user

```bash
xme search payments-api "auth decisions"          # search team memory
xme facts payments-api --type decision            # list team decisions
xme facts payments-api --type preference --user raj  # personal preferences
```

---

## Export formats

**Obsidian vault:**
```bash
xme export my-project --format obsidian
# → .xanther/obsidian/  (open as Obsidian vault)
```

**Agent-navigable wiki:**
```bash
xme export my-project --format wiki
# → .xanther/wiki/index.md + article per concept
```

**Graphify-compatible:**
```bash
xme export my-project --format graphify
# → .xanther/graphify-out/graph.json + GRAPH_REPORT.md
```

---

## Storage backends

| Backend | What it stores | Required? |
|---------|---------------|-----------|
| SQLite | Context layer + fact fallback + episodic fallback | Always (built-in) |
| Neo4j | Fact graph with vector search | Recommended |
| OpenSearch | Episodic full-text + semantic search | Optional |

**Zero-infrastructure mode** (SQLite only):
```bash
XME_FALLBACK_MODE=true xme start my-project
```

**Full mode** (Neo4j + OpenSearch via Docker):
```bash
docker-compose up -d
xme start my-project
```

---

## Comparison

| Feature | Graphify | Mem0 | Zep | **Xanther XME** |
|---------|----------|------|-----|-----------------|
| Code knowledge graph | ✅ | ❌ | ❌ | ✅ (via XCE) |
| Session memory | ❌ | ✅ | ✅ | ✅ |
| Fact graph | ❌ | partial | ✅ | ✅ |
| Working context UPSERT | ❌ | ❌ | ❌ | ✅ |
| Multi-user scoping | ❌ | ✅ | ✅ | ✅ |
| Deduplication | ❌ | ✅ | ✅ | ✅ |
| Local-first / self-hosted | ✅ | ❌ | ❌ | ✅ |
| No pre-indexing required | ✅ | ✅ | ✅ | ✅ |
| Obsidian export | ✅ | ❌ | ❌ | ✅ |
| Dashboard UI | graph.html | ❌ | ✅ | ✅ |
| MCP tools | 1 | ❌ | ❌ | 11 |
| Open source | ✅ | partial | ❌ | ✅ |

---

## Architecture

```
xme/
├── engine.py          # MemoryEngine — single entry point
├── models.py          # Episode, Fact, WorkingContext, Turn
├── config.py          # XMESettings (env-based)
├── layers/
│   ├── episodic.py    # OpenSearch + SQLite FTS5 fallback
│   ├── facts.py       # Neo4j + vector dedup + SQLite fallback
│   └── context.py     # SQLite UPSERT per (project, user)
├── extraction/
│   ├── embedder.py    # all-MiniLM-L6-v2 (local, no API key)
│   └── extractor.py   # LLM or regex fact extraction
├── server/
│   ├── mcp_tools.py   # 11 MCP tools
│   └── dashboard.py   # FastAPI dashboard (port 8001)
├── export/
│   ├── obsidian.py
│   ├── wiki.py
│   └── graphify_compat.py
└── hooks/
    ├── installer.py   # xme hook install
    └── handler.py     # hook entrypoint (zero imports, <20ms)
```

XCE (Xanther Context Engine) — code graph intelligence — lives alongside XME in the same package. When XCE has indexed your codebase, facts automatically link to the relevant AST nodes, enabling queries like "which decisions affected the auth module?" XME works without XCE.

---

## Configuration

All configuration via environment variables:

```bash
# Storage
XME_SQLITE_PATH=.xanther/xme.db       # default
XME_OPENSEARCH_URL=http://localhost:9200
XME_FALLBACK_MODE=false                # true = SQLite only

# Embedding
XME_EMBEDDING_MODEL=all-MiniLM-L6-v2  # local model
XME_DEDUP_THRESHOLD=0.85               # similarity threshold for merging facts

# LLM extraction (optional — better fact quality)
OPENROUTER_API_KEY=sk-...
XME_LLM_MODEL=openai/gpt-4o-mini

# Neo4j (shared with XCE)
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your-password
```

---

## CLI reference

```bash
xme start <project_id>               # init + show stats
xme add <project_id> <user> <text>   # add content to memory
xme search <project_id> <query>      # search all layers
xme facts <project_id>               # list facts
xme stats <project_id>               # memory health
xme export <project_id>              # export (obsidian/wiki/graphify)
xme dashboard                        # launch web UI (port 8001)
xme hook install [path]              # install Kiro + Claude Code hooks
xme hook uninstall [path]            # remove hooks
```

---

## Contributing

Xanther is open source under the Apache 2.0 license. Contributions welcome.

The most useful contributions:
- **Real-world sessions** — run XME on your project, share what facts got extracted (and what got missed)
- **Extraction improvements** — better regex patterns or LLM prompts for fact extraction
- **New exporters** — Notion, Linear, Confluence
- **Language support** — more tree-sitter parsers for XCE

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
