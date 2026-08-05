---
name: xme
description: Set up Xanther Memory Engine (XME) — persistent memory across AI coding sessions
trigger: /xme
---

# XME Setup Skill

When the user types `/xme`, execute this skill to set up Xanther Memory Engine step by step.
XME gives your AI assistant persistent memory — decisions remembered, failures not repeated, context always current.

## Instructions

Follow these steps IN ORDER. Do not skip any step.

---

### STEP 1 — Detect environment

```bash
python3 --version    # need 3.10+
docker --version     # optional — needed for Neo4j + OpenSearch
```

If Python < 3.10: tell user to upgrade. Do not proceed.
Docker is optional — XME runs in SQLite-only fallback mode without it.

---

### STEP 2 — Install XME

```bash
pip install xanther-memory-engine
```

Verify:
```bash
xme --help
```

If `xme` not found: `python -m xme.cli --help`

---

### STEP 3 — Choose infrastructure mode

Ask the user:
> "Do you want full mode (Neo4j + OpenSearch via Docker, recommended) or zero-infrastructure mode (SQLite only, works immediately)?"

**Zero-infrastructure mode (answer: no/simple/now):**
```bash
export XME_FALLBACK_MODE=true
xme start $(basename $(pwd))
```
→ Skip to STEP 6

**Full mode (answer: yes/full/docker):**
Continue to STEP 4.

---

### STEP 4 — Start infrastructure (full mode)

```bash
curl -fsSL https://raw.githubusercontent.com/Xanther-Ai/xanther-memory-engine/main/docker-compose.yml -o docker-compose.xme.yml
docker compose -f docker-compose.xme.yml up -d
```

Wait for services:
```bash
until curl -s http://localhost:9200 >/dev/null 2>&1; do echo "Waiting for OpenSearch..."; sleep 3; done
echo "✓ OpenSearch ready"
until docker exec $(docker ps -qf "name=neo4j") cypher-shell -u neo4j -p xme_dev_password "RETURN 1" 2>/dev/null; do echo "Waiting for Neo4j..."; sleep 3; done
echo "✓ Neo4j ready"
```

---

### STEP 5 — Configure environment

Create `.env` in current directory if it doesn't exist:

```bash
cat > .env << 'EOF'
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=xme_dev_password
XME_OPENSEARCH_URL=http://localhost:9200
XME_FALLBACK_MODE=false
XME_DEDUP_THRESHOLD=0.85
EOF
```

Ask user: "Do you have an OpenRouter API key for smarter fact extraction? (optional — regex extraction works without it)"

If yes:
```bash
echo "OPENROUTER_API_KEY=<their-key>" >> .env
echo "XME_LLM_MODEL=openai/gpt-4o-mini" >> .env
```

---

### STEP 6 — Initialize XME for this project

```bash
PROJECT_ID=$(basename $(pwd))
xme start $PROJECT_ID
```

Expected output:
```
✓ XME initialized for 'my-project'
  DB: .xanther/xme.db
  Facts:    0
  Episodes: 0
```

---

### STEP 7 — Install IDE hooks

This makes XME auto-capture every session without any manual action:

```bash
xme hook install .
```

Expected output:
```
✓ Installed XME hooks:
  kiro: .kiro/hooks/xme-session-end.kiro.hook
  kiro: .kiro/hooks/xme-record-turn.kiro.hook
  claude: .claude/settings.json
```

Verify hooks were created:
```bash
ls .kiro/hooks/ | grep xme
cat .claude/settings.json | python3 -m json.tool | grep xme
```

---

### STEP 8 — Configure MCP tools

Add XME MCP tools to your IDE config.

**For Claude Code** — add to `~/.claude/settings.json` (merge with existing):
```json
{
  "mcpServers": {
    "xme": {
      "command": "xme",
      "args": ["serve"],
      "env": {
        "NEO4J_PASSWORD": "xme_dev_password",
        "XME_FALLBACK_MODE": "false"
      }
    }
  }
}
```

**For Kiro** — add to `~/.kiro/settings/mcp.json` with same structure.

Tell user to restart their IDE after adding config.

---

### STEP 9 — Add first memory

Test that XME is working by adding a fact:

```bash
PROJECT_ID=$(basename $(pwd))
xme add $PROJECT_ID $(whoami) "XME is now set up and capturing memory for this project"
```

Then verify:
```bash
xme facts $PROJECT_ID
```

Should show 1 fact of type `entity`.

---

### STEP 10 — Test session start

In Claude Code / Kiro, run this MCP call to confirm end-to-end:

```
Use xme_session_start with project_id="<project-name>" and user_id="<your-name>"
```

It should return a `prompt_block` with your context (empty on first run, populated after sessions).

---

### STEP 11 — Verify hooks are firing

Send a test message, then check the buffer:

```bash
ls .xanther/turns/ 2>/dev/null && echo "✓ hooks writing buffer files" || echo "No buffer files yet — send a message first"
```

---

## Usage after setup

**Search your memory:**
```bash
xme search my-project "auth decisions"
xme facts my-project --type decision
xme stats my-project
```

**Launch dashboard:**
```bash
xme dashboard
# Open http://localhost:8001
```

**Export to Obsidian:**
```bash
xme export my-project --format obsidian
```

**Agent workflow** (automatic once hooks are installed):
- Every prompt → buffered to `.xanther/turns/`
- Every `agentStop` → buffer drains → facts extracted → context updated
- Next session → `xme_session_start` returns primed context block

---

## Troubleshooting

**"xme command not found"** → `pip install xanther-memory-engine` and check PATH
**"Neo4j connection refused"** → `docker compose -f docker-compose.xme.yml ps`; or set `XME_FALLBACK_MODE=true`
**Hooks not firing** → Check `.kiro/hooks/xme-*.kiro.hook` exist and `enabled: true`
**Empty facts after sessions** → Hooks write buffers; run `xme start <project>` to drain them manually
**Slow embedding** → First run downloads `all-MiniLM-L6-v2` (~80MB). Subsequent runs are instant.

## What you get

After setup, every session is automatically captured:

```
Session 1: "We decided to use FastAPI. Redis lock failed — timeout under load."
Session 2: Agent already knows. Doesn't suggest Redis. Doesn't re-explain FastAPI.
Session 10: Full institutional memory. 
```

Memory tools available via MCP: `xme_session_start/end`, `xme_add`, `xme_search`, `xme_get_context`, `xme_facts`, `xme_episodes`, `xme_remember`, `xme_forget`, `xme_export`, `xme_context_update`

GitHub: https://github.com/Xanther-Ai/xanther-memory-engine
