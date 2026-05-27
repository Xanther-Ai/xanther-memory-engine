-- XME Initial Schema
-- PostgreSQL 16 + TimescaleDB

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- ============================================================
-- raw_messages: Ingested conversation messages from MCP clients
-- ============================================================
CREATE TABLE IF NOT EXISTS raw_messages (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      VARCHAR(255) NOT NULL,
    user_id         VARCHAR(255) NOT NULL,
    repo_id         VARCHAR(255) NOT NULL,
    role            VARCHAR(50) NOT NULL,
    content         TEXT,
    tool_calls      JSONB,
    processed       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for worker polling and queries
CREATE INDEX idx_raw_messages_processed ON raw_messages (processed) WHERE processed = FALSE;
CREATE INDEX idx_raw_messages_session_id ON raw_messages (session_id);
CREATE INDEX idx_raw_messages_repo_id ON raw_messages (repo_id);
CREATE INDEX idx_raw_messages_created_at ON raw_messages (created_at DESC);

-- Convert to TimescaleDB hypertable for time-series performance
SELECT create_hypertable('raw_messages', 'created_at', if_not_exists => TRUE);

-- ============================================================
-- extracted_memories: Distilled knowledge from raw messages
-- ============================================================
CREATE TABLE IF NOT EXISTS extracted_memories (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id          VARCHAR(255) NOT NULL,
    user_id             VARCHAR(255) NOT NULL,
    repo_id             VARCHAR(255) NOT NULL,
    kind                VARCHAR(50) NOT NULL,
    summary             TEXT NOT NULL,
    reasoning           TEXT,
    confidence          FLOAT NOT NULL DEFAULT 0.8,
    priority            INT NOT NULL DEFAULT 2,
    refs                JSONB,
    source_message_ids  UUID[],
    written_to_graph    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for worker and query patterns
CREATE INDEX idx_extracted_memories_written ON extracted_memories (written_to_graph) WHERE written_to_graph = FALSE;
CREATE INDEX idx_extracted_memories_session_id ON extracted_memories (session_id);
CREATE INDEX idx_extracted_memories_repo_id ON extracted_memories (repo_id);
CREATE INDEX idx_extracted_memories_kind ON extracted_memories (kind);
CREATE INDEX idx_extracted_memories_created_at ON extracted_memories (created_at DESC);
CREATE INDEX idx_extracted_memories_user_repo ON extracted_memories (user_id, repo_id);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('extracted_memories', 'created_at', if_not_exists => TRUE);

-- ============================================================
-- session_state: Lightweight key-value session state
-- ============================================================
CREATE TABLE IF NOT EXISTS session_state (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id  VARCHAR(255) NOT NULL,
    user_id     VARCHAR(255) NOT NULL,
    repo_id     VARCHAR(255),
    key         VARCHAR(255) NOT NULL,
    value       JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(session_id, key)
);

CREATE INDEX idx_session_state_session ON session_state (session_id);
CREATE INDEX idx_session_state_user ON session_state (user_id);
