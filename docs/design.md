# Design: Session Memory Sync to XCE Graph

## Problem Statement

AI coding agents (Claude Code, Cursor, Kiro, etc.) generate session memories during development sessions — logs of file edits, command executions, code snippets, decisions, and reasoning. Currently, this context is lost between sessions.

**Goal:** Sync session memories to XCE's knowledge graph in a way that:
1. Enriches the graph with agent context (decisions, changes, learnings)
2. Doesn't explode graph size (selective, compressed storage)
3. Works offline (queued sync)
4. Syncs to mobile (push notifications + local cache)

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  AI Agent       │     │  Session Memory  │     │  XCE Graph      │
│  (Claude/Cursor │────▶│  Buffer          │────▶│  (Neo4j)        │
│   Kiro)         │     │  (Local SQLite)  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                         ▲
                               ▼                         │
                        ┌──────────────────┐            │
                        │  Sync Engine     │◀───────────┘
                        │  (Dedupe/Compress│
                        │   Priority Queue)│
                        └──────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ Cloud Sync  │     │ Mobile Push │     │ Graph       │
   │ (REST API)  │     │ (FCM/APNs)  │     │ Updates     │
   └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Core Concepts

### 1. Session Memory Types

| Type | Description | Graph Impact |
|------|-------------|--------------|
| `file_edit` | File modified, lines added/removed | Low (links to existing nodes) |
| `code_decision` | Why a particular approach was taken | Medium (new decision nodes) |
| `bug_discovered` | Bug found during session | Medium (new issue nodes) |
| `refactor_done` | Code refactored | High (creates refactor edge) |
| `api_call` | External API used | Low (links to existing nodes) |
| `test_written` | Test file created/modified | Low (links to code nodes) |
| `question_asked` | Developer asked agent a question | High (new QA nodes) |
| `answer_given` | Agent responded with context | High (new context nodes) |

### 2. Memory Node Kinds (New)

```python
class MemoryKind(str, Enum):
    SESSION = "session"           # A coding session
    DECISION = "decision"         # Architectural/design decision
    QUESTION = "question"         # Developer question
    ANSWER = "answer"             # Agent response  
    BUG = "bug"                   # Bug discovered
    REFACTOR = "refactor"         # Refactoring action
    INSIGHT = "insight"           # Code insight/learnings
    CONTEXT = "context"           # Retrieved context used
```

### 3. Memory Edge Types

```python
class MemoryEdge(str, Enum):
    BELONGS_TO_SESSION = "belongs_to_session"
    DECISION_FOR = "decision_for"        # Decision relates to code
    ANSWERS = "answers"                  # Answer addresses question
    CONTEXT_FOR = "context_for"          # Context used for answer
    FIXES_BUG = "fixes_bug"              # Fix addresses bug
    REFS = "refs"                        # References existing code node
```

---

## Sync Strategy: The "Don't Explode the Graph" Rules

### Rule 1: Separate Label Namespace (Option B)
- Memory nodes use `:Memory` label, code nodes use `:ASTNode`
- XCE queries ONLY match `(:ASTNode)` — never see memory nodes
- XME queries start from `(:Memory)` and optionally follow `REFS` edges to code
- One-way edges: Memory → Code (never Code → Memory)

### Rule 2: Hard Cap Per Repo (Option C)
- Maximum **500 memory nodes per repo**
- When cap is reached, evict lowest-confidence + oldest memories
- Eviction rules:
  1. If referenced code node was deleted → evict immediately
  2. If confidence < 0.7 and age > 30 days → evict
  3. If 3+ similar decisions exist → merge into one canonical
  4. Oldest low-priority memories evicted first

### Rule 3: Link vs Create
- **ALWAYS link to existing graph nodes** (functions, classes, modules)
- **ONLY create new nodes** when truly novel (new decisions, bugs, questions)
- Never duplicate existing code structure

### Rule 2: Temporal Pruning
- Keep last 30 days of memories in full
- Compress 30-90 days to weekly summaries
- Keep 90+ days as monthly aggregates

### Rule 3: Priority Queue
Not all memories are equal. Prioritize:

| Priority | Memory Type | Sync Delay | Retention |
|----------|-------------|------------|-----------|
| P0 (urgent) | Bug fixes, security issues | Immediate | Forever |
| P1 (high) | Important decisions, architecture changes | < 1 hour | Forever |
| P2 (medium) | Code edits, refactors | < 24 hours | 90 days |
| P3 (low) | Minor edits, questions | < 7 days | 30 days |

### Rule 4: Semantic Deduplication
- If same decision is recorded multiple times, keep latest only
- Use embeddings to detect near-duplicates (cosine similarity > 0.9)
- Merge similar questions into canonical form

### Rule 5: Compression
- Store code diffs, not full files
- Compress text with tokenization
- Store embeddings for semantic search, not full text

---

## Data Flow

### Phase 1: Local Collection (Agent Side)

```typescript
// In the AI agent - captures memory during session
interface SessionMemory {
  id: string;                      // UUID
  session_id: string;              // Session identifier
  timestamp: number;               // Unix timestamp
  kind: MemoryKind;
  priority: 0 | 1 | 2 | 3;
  
  // What happened
  content: string;                 // The memory text
  code_snippet?: string;           // Relevant code (truncated)
  diff?: string;                   // Unified diff if file edit
  
  // Links to graph (existing nodes)
  references: {                    // Existing nodes this references
    node_id: string;
    node_kind: string;             // function, class, module
  }[];
  
  // New entities discovered (will create nodes)
  new_entities?: {
    name: string;
    kind: "decision" | "bug" | "question" | "insight";
    description: string;
  }[];
  
  // Sync metadata
  synced: boolean;
  retry_count: number;
}
```

### Phase 2: Local Buffer (SQLite on Agent)

```python
# Local SQLite database
class SessionMemoryDB:
    # Queue of unsynced memories
    unsynced: List[SessionMemory]  # Ordered by priority
    
    # Recently synced (for dedup)
    recent_hashes: Set[str]        # SHA256(content + timestamp)
    
    # Compression cache
    embeddings: Dict[str, List[float]]  # Cached embeddings
    
    def add(memory: SessionMemory):
        # Check dedup
        if hash(memory.content) in recent_hashes:
            return
        # Add to queue by priority
        unsynced.insert(memory.priority, memory)
        
    def get_batch(limit: int) -> List[SessionMemory]:
        # Get highest priority items
        return unsynced[:limit]
        
    def mark_synced(ids: List[str]):
        # Remove from queue, add to recent
        for id in ids:
            memory = find(id)
            recent_hashes.add(hash(memory.content))
            unsynced.remove(memory)
```

### Phase 3: Sync Engine (Server Side)

```python
class SyncEngine:
    def process_batch(memories: List[SessionMemory]):
        # 1. Deduplicate globally
        memories = self.global_dedup(memories)
        
        # 2. Link to existing graph nodes
        for mem in memories:
            mem.references = self.resolve_references(mem)
            
        # 3. Apply priority rules
        self.apply_retention_policy(memories)
        
        # 4. Batch write to Neo4j
        self.write_to_graph(memories)
        
        # 5. Trigger mobile notifications
        self.notify_mobile(memories)
        
    def resolve_references(memory: SessionMemory) -> List[str]:
        """Find existing graph nodes that match this memory's references"""
        # Use XCE search to find matching functions/classes
        # Return their node IDs
        node_ids = []
        for ref in memory.references:
            matches = xce.search(ref.name, limit=1)
            if matches:
                node_ids.append(matches[0].id)
        return node_ids
        
    def apply_retention_policy(memories: List[SessionMemory]):
        """Apply temporal pruning rules"""
        now = time.time()
        thirty_days = 30 * 24 * 3600
        
        for mem in memories:
            age = now - mem.timestamp
            if age > thirty_days * 3:
                mem.priority = 3  # Lowest priority
                mem.content = compress(mem.content)  # Compress
```

### Phase 4: Graph Storage (Neo4j)

```cypher
// Memory nodes are separate from code nodes
// They reference code nodes but don't pollute them

// Create memory node
CREATE (m:Memory {
  id: $id,
  session_id: $session_id,
  kind: $kind,
  content: $content,
  timestamp: $timestamp,
  priority: $priority
})

// Link to existing code nodes (don't create duplicates)
MATCH (c:ASTNode)
WHERE c.id IN $references
CREATE (m)-[:REFS]->(c)

// Link to session
MATCH (s:Session {id: $session_id})
CREATE (s)-[:HAS_MEMORY]->(m)

// For decisions, link to the code they relate to
IF $kind = 'decision':
  MATCH (c:ASTNode {id: $target_node})
  CREATE (m)-[:DECISION_FOR]->(c)
```

---

## Mobile Sync Design

### Push Notification Strategy

When new memories are synced to graph, mobile needs to know about:

1. **Important decisions** affecting their codebase
2. **Bugs discovered** that might impact them
3. **Context retrieved** for their questions (answers they got)

### Mobile Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  XCE Graph      │     │  Push Service    │     │  Mobile App     │
│  (Neo4j)        │────▶│  (FCM/APNs)      │────▶│  (Notifications)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                                                │
        │                                                ▼
        │                                       ┌─────────────────┐
        └──────────────────────────────────────▶│  Local Cache    │
                                                │  (SQLite)       │
                                                └─────────────────┘
```

### Push Payload

```json
{
  "notification": {
    "title": "New context from your coding session",
    "body": "Decision: Chose async over sync for API calls in utils/http.py"
  },
  "data": {
    "type": "decision",
    "memory_id": "mem_123",
    "session_id": "sess_456",
    "references": ["node_789"],  // Code nodes affected
    "priority": "1"
  }
}
```

### Mobile Local Cache

```typescript
// Mobile SQLite schema
interface MobileMemoryCache {
  memories: {
    id: string;
    session_id: string;
    kind: string;
    content: string;
    timestamp: number;
    is_read: boolean;
  }[];
  
  code_references: {    // Cached graph nodes for offline
    node_id: string;
    name: string;
    kind: string;
    filepath: string;
  }[];
}
```

### Mobile Hooks (React Query style)

```typescript
// Hook: Subscribe to new memories for a repo
function useRepoMemories(repoId: string) {
  // Real-time subscription via WebSocket or polling
  const { data, isLoading } = useQuery({
    queryKey: ['memories', repoId],
    queryFn: () => api.getMemories(repoId),
    // Subscribe to push notifications
    subscribe: (onNew) => pushService.subscribe(
      `repo:${repoId}`, 
      onNew
    )
  });
  
  return { memories: data ?? [], isLoading };
}

// Hook: Get memory details with linked code
function useMemoryDetail(memoryId: string) {
  return useQuery({
    queryKey: ['memory', memoryId],
    queryFn: () => api.getMemoryWithReferences(memoryId),
    // Prefetch linked code nodes
    prefetch: (memory) => {
      memory.references.forEach(ref => {
        queryClient.prefetchQuery(['node', ref.node_id], ...)
      })
    }
  });
}

// Hook: Mark memory as read (syncs back)
function useMarkMemoryRead() {
  const mutation = useMutation({
    mutationFn: (memoryId: string) => 
      api.markRead(memoryId),
    onSuccess: () => {
      // Update local cache immediately
      queryClient.invalidateQueries(['memories'])
    }
  });
  
  return mutation.mutate;
}

// Hook: Offline queue for memories created on mobile
function useOfflineMemoryQueue() {
  const queue = useRef<SessionMemory[]>([]);
  
  function addMemory(memory: SessionMemory) {
    queue.current.push(memory);
    // Try to sync when online
    if (navigator.onLine) {
      syncQueue();
    }
  }
  
  async function syncQueue() {
    while (queue.current.length > 0) {
      const batch = queue.current.splice(0, 10);
      try {
        await api.syncMemories(batch);
      } catch (e) {
        // Put back in queue
        queue.current.unshift(...batch);
        break;
      }
    }
  }
  
  return { addMemory, queue: queue.current };
}
```

---

## Hooks API Summary

### Agent Side Hooks

```typescript
// Capture memory during agent session
function useSessionMemory() {
  const sessionId = useRef(uuid());
  
  // Call when agent makes a decision
  function recordDecision(content: string, references: NodeRef[]) {
    const memory: SessionMemory = {
      id: uuid(),
      session_id: sessionId.current,
      timestamp: Date.now(),
      kind: 'decision',
      priority: determinePriority(references),
      content,
      references,
      synced: false
    };
    localDB.add(memory);
  }
  
  // Call when file is edited
  function recordEdit(file: string, diff: string) {
    const memory: SessionMemory = {
      id: uuid(),
      session_id: sessionId.current,
      timestamp: Date.now(),
      kind: 'file_edit',
      priority: 2,
      content: `Edited ${file}`,
      diff,
      references: [{ node_id: file, node_kind: 'file' }],
      synced: false
    };
    localDB.add(memory);
  }
  
  return { recordDecision, recordEdit };
}
```

### Mobile Side Hooks

```typescript
// Core hooks for mobile app
export const useMemories = {
  // Subscribe to repo memories (real-time)
  useSubscribe(repoId: string): Observable<Memory[]>
  
  // Get single memory with code context
  useDetail(memoryId: string): MemoryWithRefs
  
  // Create memory offline (queued for sync)
  useCreate(): (memory: MemoryInput) => void
  
  // Mark as read
  useMarkRead(): (memoryId: string) => void
  
  // Search memories (offline capable)
  useSearch(query: string, repoId: string): Memory[]
}
```

---

## Summary: Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Link vs Create** | Prevents graph explosion by always referencing existing nodes |
| **Priority Queue** | Ensures important memories sync first, low-priority get batched |
| **Temporal Pruning** | Limits storage while keeping recent memories accessible |
| **Semantic Dedup** | Prevents duplicate nodes from similar memories |
| **Compression** | Reduces storage for older memories |
| **Push to Mobile** | Keeps mobile in sync with important decisions |
| **Local SQLite** | Enables offline access and queuing |
| **Hooks API** | Provides clean React-style interface for both agent and mobile |

---

## Next Steps

1. Implement local SQLite buffer in agent
2. Build sync REST API endpoint
3. Create Neo4j memory node schema
4. Implement push notification service (FCM)
5. Build mobile hooks in React Native
6. Add WebSocket for real-time updates