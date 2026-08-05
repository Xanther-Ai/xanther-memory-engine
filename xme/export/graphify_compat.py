"""Graphify-compatible export: graph.json + GRAPH_REPORT.md."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from xme.engine import MemoryEngine


async def export_graphify(
    engine: "MemoryEngine",
    project_id: str,
    output_dir: Optional[str] = None,
) -> Path:
    base = Path(output_dir or f".xanther/graphify-out/{project_id}")
    base.mkdir(parents=True, exist_ok=True)

    facts = await engine.facts.list_facts(project_id, limit=1000)
    graph = await engine.facts.get_graph_data(project_id)

    # graph.json in Graphify node format
    gj_nodes = []
    for n in graph["nodes"]:
        gj_nodes.append({
            "id": n["id"],
            "label": n["label"],
            "type": n["group"],
            "confidence": n.get("confidence", "EXTRACTED"),
            "description": n.get("title", ""),
        })
    gj_edges = []
    for e in graph["edges"]:
        gj_edges.append({
            "source": e["from"],
            "target": e["to"],
            "type": e.get("label", "RELATED_TO"),
            "confidence": "EXTRACTED",
        })

    graph_json = {"nodes": gj_nodes, "edges": gj_edges, "metadata": {"project_id": project_id}}
    (base / "graph.json").write_text(json.dumps(graph_json, indent=2))

    # GRAPH_REPORT.md
    # Find god nodes (highest degree)
    from collections import Counter
    degree: Counter = Counter()
    for e in gj_edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    id_to_label = {n["id"]: n["label"] for n in gj_nodes}
    god_nodes = degree.most_common(10)

    type_counts: Counter = Counter(n["type"] for n in gj_nodes)

    report = [
        f"# Graph Report — {project_id}",
        "",
        f"**Nodes**: {len(gj_nodes)}  **Edges**: {len(gj_edges)}",
        "",
        "## Node Types",
        "",
    ]
    for t, c in type_counts.most_common():
        report.append(f"- {t}: {c}")
    report += ["", "## God Nodes (highest connectivity)", ""]
    for nid, deg in god_nodes:
        report.append(f"- **{id_to_label.get(nid, nid)}** — {deg} connections")
    report += ["", "## Suggested Questions", ""]
    decision_labels = [id_to_label[n["id"]] for n in gj_nodes if n["type"] == "decision"][:3]
    for dl in decision_labels:
        report.append(f"- What led to the decision: {dl}?")
    report += [
        "- What are the most contested decisions?",
        "- Which attempts failed and why?",
        "- What conventions does this team follow?",
    ]
    (base / "GRAPH_REPORT.md").write_text("\n".join(report))

    return base
