"""XME export — Obsidian vault, wiki markdown, Graphify-compatible JSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from xme.engine import MemoryEngine

logger = logging.getLogger(__name__)


async def run_export(
    engine: "MemoryEngine",
    project_id: str,
    fmt: str = "obsidian",
    output_dir: Optional[str] = None,
) -> Path:
    if fmt == "obsidian":
        from xme.export.obsidian import export_obsidian
        return await export_obsidian(engine, project_id, output_dir)
    elif fmt == "wiki":
        from xme.export.wiki import export_wiki
        return await export_wiki(engine, project_id, output_dir)
    elif fmt == "graphify":
        from xme.export.graphify_compat import export_graphify
        return await export_graphify(engine, project_id, output_dir)
    else:
        raise ValueError(f"Unknown export format: {fmt}")
