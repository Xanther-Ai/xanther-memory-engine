<div align="center">

# Xanther Memory Engine (XME)

**Persistent memory for AI coding assistants.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/xanther-xme)](https://pypi.org/project/xanther-xme)

</div>

---

> Your AI assistant forgets every decision you've made. It repeats the same failed approaches. It re-explains your stack every session. XME fixes this.

XME gives AI coding assistants **persistent memory** across sessions. Works with Claude Code, Kiro, Cursor, Codex, and any MCP-compatible tool. No cloud required.

```bash
pip install xanther-xme
xme hook install .      # 30 seconds — auto-captures every session
xme start my-project    # memory starts now
```

> **Want code intelligence too?** Install XME bundled with the [Xanther Context Engine (XCE)](https://github.com/Xanther-Ai/xanther-context-engine) in one command:
> ```bash
> pip install "xanther-xce[all]"        # XCE + XME together
> # or run instantly, no install:
> uvx --from "xanther-xce[all]" xanther --help
> ```

---

## Architecture

```mermaid
graph TB
    subgraph "AI Agent (Claude Code / Kiro / Cursor)"
        AGENT[Agent]
        HOOKS[IDE Hooks<br/>agentStop · promptSubmit]
    end

    subgraph "XME Memory Engine"
        ENGINE[MemoryEngine<br/>xme/engine.py]

        subgraph "Layer 1 — Episodic"
            EP[EpisodicStore<br/>Verbatim session transcripts]
        end

        subgraph "Layer 2 — Facts"
            FG[FactGraphStore<br/>Decisions · Attempts<br/>Preferences · Conventions]
            EXT[FactExtractor<br/>LLM or regex]
            EMB[LocalEmbedder<br/>all-MiniLM-L6-v2]
            EXT --> FG
            EMB --> FG
        end

        subgraph "Layer 3 — Context"
            CTX[ContextStore<br/>Working state per project+user<br/>UPSERT semantics]
        end

        ENGINE --> EP & FG & CTX
    end

    subgraph "Storage"
        OS[(OpenSearch<br/>port 9200<br/>Full-text + k-NN)]
        NEO4J[(Neo4j<br/>port 7687<br/>Fact graph + vectors)]
        SQLITE[(SQLite<br/>.xanther/xme.db<br/>Context + fallback)]
    end

    subgraph "Outputs"
        MCP[MCP Server<br/>11 tools]
        DASH[Dashboard<br/>port 8001]
        EXP[Exports<br/>Obsidian · Wiki · Graphify]
    end

    HOOKS -- buffer files --> ENGINE
    AGENT -- MCP tool calls --> MCP
    EP --> OS & SQLITE
    FG --> NEO4J & SQLITE
    CTX --> SQLITE
    ENGINE --> DASH & EXP
    ENGINE --> MCP
```

---

## Local Infrastructure

```mermaid
graph LR
    subgraph "Your Machine"
        subgraph "Docker Compose"
            NEO4J[(Neo4j:7687<br/>Fact knowledge graph)]
            OS[(OpenSearch:9200<br/>Episodic search)]
        end

        subgraph "XME Process"
            CLI[xme CLI]
            DASH[xme dashboard<br/>:8001]
            MCP_SRV[MCP Server]
        end

        subgraph "Hook Files"
            BUF[.xanther/turns/<br/>Buffer files<br/>written per turn]
            DB[.xanther/xme.db<br/>SQLite warm store]
        end

        subgraph "IDE"
            KIRO[Kiro / Claude Code]
            MCP_CFG[mcp.json]
        end
    end

    subgraph "External APIs (optional)"
        OR[OpenRouter API<br/>LLM fact extraction]
    end

    KIRO -- agentStop hook --> BUF
    KIRO -- promptSubmit hook --> BUF
    CLI -- drain buffer --> DB
    CLI -- index to --> NEO4J & OS
    MCP_CFG -- spawn --> MCP_SRV
    MCP_SRV -- read --> NEO4J & OS & DB
    KIRO -- MCP tool calls --> MCP_SRV
    CLI -. LLM extraction .-> OR
    DASH -- read --> NEO4J & OS & DB
```

---

## Session lifecycle

```mermaid
sequenceDiagram
    participant IDE as Kiro / Claude Code
    participant HOOK as Hook Handler<br/>.xanther/hook.py
    participant BUF as Buffer<br/>.xanther/turns/
    participant XME as XME Engine
    participant DB as Neo4j + SQLite

    IDE->>HOOK: promptSubmit (user message)
    HOOK->>BUF: write turn JSON (< 5ms)

    IDE->>HOOK: promptSubmit (next message)
    HOOK->>BUF: write turn JSON

    Note over IDE,DB: ... more turns ...

    IDE->>HOOK: agentStop (response finished)
    HOOK->>BUF: write session_end marker

    Note over BUF,DB: On next xme start or xme_session_end MCP call

    XME->>BUF: drain all buffer files
    XME->>XME: extract facts (LLM or regex)
    XME->>DB: upsert facts with vector dedup
    XME->>DB: save episode to OpenSearch
    XME->>DB: update working context (UPSERT)

    Note over IDE,DB: Next session

    IDE->>XME: xme_session_start
    XME->>DB: load working context
    XME->>DB: load recent facts
    XME->>DB: load last episode summary
    XME-->>IDE: primed context block (inject into prompt)
```

---

## Three memory layers

```mermaid
flowchart LR
    subgraph "Layer 1 — Episodic"
        direction TB
        E1[Full session transcripts<br/>verbatim]
        E2[Searchable by:<br/>full-text · semantic · date · user]
        E3[Backend: OpenSearch<br/>Fallback: SQLite FTS5]
        E1 --> E2 --> E3
    end

    subgraph "Layer 2 — Facts"
        direction TB
        F1[Extracted knowledge nodes]
        F2[Types:<br/>Decision · Attempt<br/>Preference · Convention · Entity]
        F3[UPSERT dedup<br/>cosine similarity > 0.85]
        F4[Backend: Neo4j graph<br/>+ vector index]
        F1 --> F2 --> F3 --> F4
    end

    subgraph "Layer 3 — Context"
        direction TB
        C1[Live working state<br/>per project + user]
        C2[Fields:<br/>current_task · next_steps<br/>recent_decisions · blockers]
        C3[UPSERT only — always current<br/>Backend: SQLite]
        C1 --> C2 --> C3
    end

    EP[Episodic\nStore] --> L1(Layer 1)
    FG[Fact\nGraph] --> L2(Layer 2)
    CTX[Context\nStore] --> L3(Layer 3)

    style L1 fill:#dbeafe
    style L2 fill:#dcfce7
    style L3 fill:#fef9c3
```

---

## Getting Started

### 1. Install

```bash
pip install xanther-xme

# Or bundled with the Xanther Context Engine (code graph + memory):
pip install "xanther-xce[all]"
# Run instantly without installing:
uvx --from "xanther-xce[all]" xanther --help
```

### 2. Choose an infrastructure mode

XME works with or without Docker. Pick one:

**Zero infrastructure** — SQLite only, no Docker, works offline:
```bash
XME_FALLBACK_MODE=true xme start my-project
```

**Full infrastructure** — Neo4j (fact graph) + OpenSearch (episodic search):
```bash
cp .env.example .env        # set NEO4J_PASSWORD (and OPENROUTER_API_KEY for LLM extraction)
docker-compose up -d        # starts Neo4j (:7687) + OpenSearch (:9200)
xme start my-project
```

> No OpenRouter key? Fact extraction falls back to regex heuristics — everything still works,
> just with slightly coarser facts.

### 3. Install the auto-capture hooks

This is the step that makes memory automatic. Hooks capture every agent turn and persist a
session when the agent stops — no manual recording needed.

```bash
# Install Kiro + Claude Code hooks into a repo (defaults to current dir)
xme hook install .

# Preview what would be written without changing anything
xme hook install . --dry-run

# Remove the hooks later
xme hook uninstall .
```

**What gets installed:**

| Hook | IDE event | What it does |
|------|-----------|--------------|
| `xme-record-turn` | `promptSubmit` | Buffers each user turn to `.xanther/turns/` (<5ms, non-blocking) |
| `xme-record-tool` | `postToolUse` | Buffers tool calls to the same journal |
| `xme-session-end` | `agentStop` / `Stop` | Drains the buffer → extracts facts → updates context → saves the session |

The installer writes IDE-native config:
- **Kiro** → hook files under `.kiro/hooks/`
- **Claude Code** → hook entries in the project's Claude settings

### 4. Wire up the MCP server (optional but recommended)

So your agent can query and prime memory directly, add XME as an MCP server:

```json
{
  "mcpServers": {
    "xme": {
      "command": "xme",
      "args": ["serve"],
      "env": { "NEO4J_PASSWORD": "your-password" }
    }
  }
}
```

At the start of a session the agent calls `xme_session_start` to get a **primed context block**
(current task, recent decisions, known-failed approaches) injected into its prompt.

### 5. Verify it's working

```bash
xme stats my-project      # memory health: fact / episode / context counts
xme dashboard             # visual timeline at http://localhost:8001
```

After a session or two, `xme stats` should show growing fact and episode counts. If they stay at
zero, see **Troubleshooting hooks** below.

### Troubleshooting hooks

- **Nothing captured?** Confirm hooks installed: check `.kiro/hooks/` (Kiro) or your Claude Code
  settings. Re-run `xme hook install . --dry-run` to see expected paths.
- **Buffer never drains?** Facts are extracted on `agentStop` or on the next `xme start` /
  `xme_session_end` MCP call. Run `xme start my-project` to force a drain.
- **Neo4j errors?** You can run fully local with `XME_FALLBACK_MODE=true` (SQLite only).
- **Buffer files** live in `.xanther/turns/`; the warm store is `.xanther/xme.db`. Both are safe
  to inspect. Add `.xanther/` to your `.gitignore` (memory is per-developer runtime state).

---

## What gets captured automatically

After `xme hook install .`:

- Every prompt is buffered to `.xanther/turns/` (< 5ms, no blocking)
- On `agentStop`: buffer drains → facts extracted → context updated
- Next session: agent gets a primed context block injected automatically

```
**Current task**: Refactor auth module
**Last session**: Moved JWT to dedicated auth service — success
**Recent decisions**:
  - [VALIDATED] Use FastAPI — async support required
  - [VALIDATED] PostgreSQL — ACID compliance
**Known failed approaches**:
  - Redis distributed lock — timeout under high load
**Next steps**: Deploy auth service to staging
```

---

## MCP tools (11)

| Tool | Description |
|------|-------------|
| `xme_session_start` | Start session, get primed context block |
| `xme_session_end` | End session: persist episode, extract facts, update context |
| `xme_add` | Add content — Mem0-style UPSERT with deduplication |
| `xme_search` | Search across all 3 layers simultaneously |
| `xme_get_context` | Get working context for prompt injection |
| `xme_facts` | Query fact graph (filter by type, user, keyword) |
| `xme_episodes` | Full-text + semantic search over past sessions |
| `xme_remember` | Explicitly store a typed fact |
| `xme_forget` | Soft-delete a memory node |
| `xme_export` | Export to Obsidian vault / wiki / Graphify JSON |
| `xme_context_update` | Partial UPSERT of working context fields |

Add to MCP config:
```json
{
  "mcpServers": {
    "xme": {
      "command": "xme",
      "args": ["serve"],
      "env": {
        "NEO4J_PASSWORD": "your-password"
      }
    }
  }
}
```

---

## Deduplication

Facts are stored once, not repeated across sessions:

```mermaid
flowchart TD
    A[New content added] --> B[Embed with\nall-MiniLM-L6-v2]
    B --> C{Similar fact exists?\ncosine > 0.85}
    C -- Yes --> D[Merge into existing fact\nupdate content + metadata]
    C -- No --> E[Create new fact node]
    D --> F[Update Neo4j + SQLite]
    E --> F
```

---

## Comparison

| | Mem0 | Zep | MemPalace | **XME** |
|--|------|-----|-----------|---------|
| Episodic memory | ✅ | ✅ | ✅ | ✅ |
| Fact graph | partial | ✅ | ❌ | ✅ |
| Working context UPSERT | ❌ | ❌ | ❌ | ✅ |
| Multi-user scoping | ✅ | ✅ | ❌ | ✅ |
| Deduplication | ✅ | ✅ | ❌ | ✅ |
| Local-first / open source | ❌ | ❌ | ✅ | ✅ |
| MCP tools | ❌ | ❌ | ❌ | ✅ (11) |
| Obsidian export | ❌ | ❌ | ❌ | ✅ |
| Dashboard UI | ❌ | ✅ | ❌ | ✅ |
| Code graph integration | ❌ | ❌ | ❌ | ✅ via XCE |

---

## CLI

```bash
xme start <project>              # init + show stats
xme add <project> <user> <text>  # add content to memory
xme search <project> <query>     # search all layers
xme facts <project>              # list facts
xme stats <project>              # memory health metrics
xme export <project>             # export (obsidian/wiki/graphify)
xme dashboard                    # launch web UI (port 8001)
xme hook install [path]          # install Kiro + Claude Code hooks
xme hook uninstall [path]        # remove hooks
```

---

## Configuration

```bash
# LLM for better fact extraction (optional — regex works without it)
OPENROUTER_API_KEY=sk-or-...
XME_LLM_MODEL=openai/gpt-4o-mini

# Neo4j — fact graph (recommended, free tier at console.neo4j.io)
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your-password

# OpenSearch — episodic search (optional, falls back to SQLite FTS5)
XME_OPENSEARCH_URL=http://localhost:9200

# Zero-infrastructure mode
XME_FALLBACK_MODE=false   # set true for SQLite-only, no Docker needed
```

See [`.env.example`](.env.example) for the complete reference.

---

## Related

[Xanther Context Engine (XCE)](https://github.com/Xanther-Ai/xanther-context-engine) — code graph intelligence. When installed alongside XME, decisions link directly to the code they affect.

```bash
pip install "xanther-context-engine[memory]"  # XCE + XME together
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).
