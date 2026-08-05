"""XME configuration — reads from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v else default

def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return default

def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v else default


@dataclass(frozen=True)
class XMESettings:
    # OpenSearch
    opensearch_url: str = field(
        default_factory=lambda: _env("XME_OPENSEARCH_URL", "http://localhost:9200")
    )
    opensearch_enabled: bool = field(
        default_factory=lambda: _env_bool("XME_OPENSEARCH_ENABLED", True)
    )

    # Neo4j (shared with XCE)
    neo4j_uri: str = field(
        default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687")
    )
    neo4j_user: str = field(
        default_factory=lambda: _env("NEO4J_USER", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: _env("NEO4J_PASSWORD", "")
    )

    @property
    def neo4j_auth(self) -> tuple[str, str]:
        return (self.neo4j_user, self.neo4j_password)

    # SQLite (context store + episodic fallback)
    sqlite_path: str = field(
        default_factory=lambda: _env("XME_SQLITE_PATH", ".xanther/xme.db")
    )

    # Local embedding model
    embedding_model: str = field(
        default_factory=lambda: _env("XME_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    embedding_dims: int = field(
        default_factory=lambda: _env_int("XME_EMBEDDING_DIMS", 384)
    )

    # LLM extraction (optional)
    llm_api_key: str = field(
        default_factory=lambda: _env("OPENROUTER_API_KEY") or _env("XME_LLM_API_KEY")
    )
    llm_model: str = field(
        default_factory=lambda: _env("XME_LLM_MODEL", "openai/gpt-4o-mini")
    )
    llm_base_url: str = field(
        default_factory=lambda: _env("XME_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    )

    # Deduplication
    dedup_threshold: float = field(
        default_factory=lambda: _env_float("XME_DEDUP_THRESHOLD", 0.85)
    )

    # Fallback mode: SQLite-only, no OpenSearch / Neo4j required
    fallback_mode: bool = field(
        default_factory=lambda: _env_bool("XME_FALLBACK_MODE", False)
    )

    # Dashboard
    dashboard_port: int = field(
        default_factory=lambda: _env_int("XME_DASHBOARD_PORT", 8001)
    )

    def resolved_sqlite_path(self, base_dir: str = ".") -> Path:
        p = Path(self.sqlite_path)
        if not p.is_absolute():
            p = Path(base_dir) / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


_settings: XMESettings | None = None


def get_settings() -> XMESettings:
    global _settings
    if _settings is None:
        _settings = XMESettings()
    return _settings


def reset_settings() -> None:
    """Force re-read from env (useful in tests)."""
    global _settings
    _settings = None
