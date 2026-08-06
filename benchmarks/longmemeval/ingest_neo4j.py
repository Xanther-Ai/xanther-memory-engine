"""
LongMemEval Neo4j Ingest — uses TemporalFactGraph for personal fact extraction.
Ingests all haystack sessions AND extracts personal facts into Neo4j graph.
"""
from __future__ import annotations
import asyncio, json, logging, sys, time
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def ingest_with_graph(
    data_path: str,
    project_id: str,
    user_id: str = "benchmark",
    limit: Optional[int] = None,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "",
    sqlite_path: str = ".xanther/bench_neo4j.db",
    resume: bool = True,
) -> dict:
    """Ingest sessions into SQLite episodic store AND Neo4j temporal fact graph."""
    import sys
    sys.path.insert(0, ".")
    from xme.config import XMESettings
    from xme.engine import MemoryEngine
    from xme.layers.temporal_graph import TemporalFactGraph, extract_personal_facts
    from xme.extraction.embedder import LocalEmbedder
    from neo4j import AsyncGraphDatabase

    questions = json.loads(Path(data_path).read_text())
    if limit:
        questions = questions[:limit]

    progress_file = Path(data_path).parent / f"ingest_neo4j_{project_id}.json"
    done_ids: set[str] = set()
    if resume and progress_file.exists():
        done_ids = set(json.loads(progress_file.read_text()).get("completed", []))
        logger.info("Resuming: %d already ingested", len(done_ids))

    # Init Neo4j
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    tfg = TemporalFactGraph(driver)
    await tfg.init_schema()

    settings = XMESettings(
        sqlite_path=sqlite_path,
        fallback_mode=True,
        opensearch_enabled=False,
    )
    embedder = LocalEmbedder()
    results = []
    start = time.time()

    async with MemoryEngine(settings) as engine:
        # Disable LLM extraction — we do our own personal fact extraction
        engine._extractor._api_key = ""

        for i, q in enumerate(questions):
            q_id = q["question_id"]
            if q_id in done_ids:
                continue

            q_project_id = f"{project_id}_{q_id}"
            sessions = q["haystack_sessions"]
            dates = q.get("haystack_dates", [])
            sess_ids = q.get("haystack_session_ids", [])
            total_facts = 0

            for j, session_turns in enumerate(sessions):
                if not session_turns:
                    continue

                date_hint = dates[j] if j < len(dates) else ""
                sid_str = sess_ids[j] if j < len(sess_ids) else f"s{j}"
                session_id = f"{q_id}_{sid_str}"

                # Build transcript
                transcript = "\n".join(
                    f"{t.get('role','user').capitalize()}: {t.get('content','')}"
                    for t in session_turns
                )

                # 1. Store in episodic (SQLite)
                ctx = await engine.session_start(q_project_id, user_id, session_id=session_id)
                for turn in session_turns:
                    engine.record_turn(ctx.session_id, turn.get("role","user"), turn.get("content","")[:500])
                await engine.session_end(ctx.session_id, q_project_id, user_id,
                                         summary=f"Session {j+1}", outcome="success")

                # 2. Extract personal facts → Neo4j
                personal_facts = extract_personal_facts(transcript, user_id, session_id, date_hint)
                for pf in personal_facts:
                    emb = embedder.embed(f"{pf['attribute']}: {pf['value']}")
                    await tfg.upsert_fact(
                        user_id=user_id,
                        attribute=pf["attribute"],
                        value=pf["value"],
                        fact_type=pf["fact_type"],
                        session_id=session_id,
                        session_date=date_hint,
                        embedding=emb,
                        project_id=q_project_id,
                    )
                    total_facts += 1

            results.append({"question_id": q_id, "sessions": len(sessions), "facts": total_facts})
            done_ids.add(q_id)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(questions) - i - 1) / rate
                logger.info("[%d/%d] ingested — %.1fs/q — ETA %.0fm — facts=%d",
                            i + 1, len(questions),
                            elapsed / (i+1), eta / 60, total_facts)
                progress_file.write_text(json.dumps({"completed": list(done_ids)}))

    await driver.close()
    progress_file.write_text(json.dumps({"completed": list(done_ids)}))
    logger.info("✓ Complete: %d questions ingested", len(results))
    return {"questions": len(results), "results": results}


if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--project-id", default="lme-neo4j")
    p.add_argument("--user-id", default="benchmark")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI","bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER","neo4j"))
    p.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD",""))
    p.add_argument("--sqlite-path", default=".xanther/bench_neo4j.db")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    r = asyncio.run(ingest_with_graph(
        data_path=args.data, project_id=args.project_id,
        user_id=args.user_id, limit=args.limit,
        neo4j_uri=args.neo4j_uri, neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password, sqlite_path=args.sqlite_path,
        resume=not args.no_resume,
    ))
    print(json.dumps(r, indent=2))
