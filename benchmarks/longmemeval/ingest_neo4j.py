"""
LongMemEval Neo4j Ingest — LLM-based personal fact extraction into TemporalFactGraph.

Ingests all haystack sessions into episodic (SQLite) AND extracts personal facts
via LLM into the Neo4j temporal graph. LLM extraction runs concurrently per session
(semaphore-bounded) for speed.
"""
from __future__ import annotations
import asyncio, json, logging, time
from pathlib import Path
from typing import Any, Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def ingest_with_graph(
    data_path: str,
    project_id: str,
    api_key: str,
    model: str = "openai/gpt-4o-mini",
    user_id: str = "benchmark",
    limit: Optional[int] = None,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "",
    sqlite_path: str = ".xanther/bench_neo4j.db",
    concurrency: int = 8,
    resume: bool = True,
) -> dict:
    """Ingest sessions into SQLite episodic store AND Neo4j temporal fact graph."""
    import sys
    sys.path.insert(0, ".")
    from xme.config import XMESettings
    from xme.engine import MemoryEngine
    from xme.layers.temporal_graph import TemporalFactGraph
    from xme.extraction.embedder import LocalEmbedder
    from xme.extraction.llm_fact_extractor import LLMFactExtractor
    from neo4j import AsyncGraphDatabase

    questions = json.loads(Path(data_path).read_text())
    if limit:
        questions = questions[:limit]

    progress_file = Path(data_path).parent / f"ingest_neo4j_{project_id}.json"
    done_ids: set[str] = set()
    if resume and progress_file.exists():
        done_ids = set(json.loads(progress_file.read_text()).get("completed", []))
        logger.info("Resuming: %d already ingested", len(done_ids))

    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    tfg = TemporalFactGraph(driver)
    await tfg.init_schema()

    settings = XMESettings(
        sqlite_path=sqlite_path,
        fallback_mode=True,
        opensearch_enabled=False,
    )
    embedder = LocalEmbedder()
    extractor = LLMFactExtractor(api_key=api_key, model=model)
    sem = asyncio.Semaphore(concurrency)
    results = []
    start = time.time()

    async with httpx.AsyncClient(timeout=45.0) as http_client:
        async with MemoryEngine(settings) as engine:
            # Disable engine's own LLM extraction — we extract personal facts ourselves
            engine._extractor._api_key = ""

            async def extract_one(transcript: str, date_hint: str):
                async with sem:
                    return await extractor.extract(transcript, date_hint, client=http_client)

            for i, q in enumerate(questions):
                q_id = q["question_id"]
                if q_id in done_ids:
                    continue

                q_project_id = f"{project_id}_{q_id}"
                sessions = q["haystack_sessions"]
                dates = q.get("haystack_dates", [])
                sess_ids = q.get("haystack_session_ids", [])
                total_facts = 0

                # Build transcripts + metadata for all sessions
                transcripts = []
                for j, session_turns in enumerate(sessions):
                    if not session_turns:
                        transcripts.append(None)
                        continue
                    date_hint = dates[j] if j < len(dates) else ""
                    sid_str = sess_ids[j] if j < len(sess_ids) else f"s{j}"
                    session_id = f"{q_id}_{sid_str}"
                    transcript = "\n".join(
                        f"{t.get('role','user').capitalize()}: {t.get('content','')}"
                        for t in session_turns
                    )
                    transcripts.append((session_id, date_hint, transcript, session_turns))

                # 1. Episodic store (sequential — SQLite)
                for entry in transcripts:
                    if entry is None:
                        continue
                    session_id, date_hint, transcript, session_turns = entry
                    ctx = await engine.session_start(q_project_id, user_id, session_id=session_id)
                    for turn in session_turns:
                        engine.record_turn(ctx.session_id, turn.get("role", "user"), turn.get("content", "")[:500])
                    await engine.session_end(ctx.session_id, q_project_id, user_id,
                                             summary=f"Session", outcome="success")

                # 2. LLM fact extraction (concurrent)
                extract_tasks = [
                    extract_one(entry[2], entry[1])
                    for entry in transcripts if entry is not None
                ]
                extracted = await asyncio.gather(*extract_tasks) if extract_tasks else []

                # 3. Upsert facts → Neo4j
                valid_entries = [e for e in transcripts if e is not None]
                for entry, facts in zip(valid_entries, extracted):
                    session_id, date_hint, _, _ = entry
                    for f in facts:
                        attribute = str(f.get("attribute", "")).strip()[:60]
                        value = str(f.get("value", "")).strip()[:200]
                        if not attribute or not value:
                            continue
                        fdate = f.get("date") or date_hint
                        # Sanitize vague dates from LLM — use session date instead
                        _VAGUE = {"today", "not specified", "recently", "current", "now",
                                  "unknown", "n/a", "none", "yesterday", "earlier", ""}
                        if str(fdate).lower().strip() in _VAGUE:
                            fdate = date_hint
                        emb = embedder.embed(f"{attribute}: {value}")
                        await tfg.upsert_fact(
                            user_id=user_id,
                            attribute=attribute,
                            value=value,
                            fact_type=str(f.get("fact_type") or attribute.lower().replace(" ", "_"))[:40],
                            session_id=session_id,
                            session_date=fdate,
                            embedding=emb,
                            project_id=q_project_id,
                        )
                        total_facts += 1

                results.append({"question_id": q_id, "sessions": len(sessions), "facts": total_facts})
                done_ids.add(q_id)

                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(questions) - i - 1) / rate
                logger.info("[%d/%d] %s — facts=%d — %.1fs/q — ETA %.1fm",
                            i + 1, len(questions), q_id, total_facts, elapsed / (i + 1), eta / 60)
                progress_file.write_text(json.dumps({"completed": list(done_ids)}))

    await driver.close()
    progress_file.write_text(json.dumps({"completed": list(done_ids)}))
    logger.info("✓ Complete: %d questions ingested", len(results))
    return {"questions": len(results), "results": results}


if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--project-id", default="lme-neo4j-llm")
    p.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--user-id", default="benchmark")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    p.add_argument("--sqlite-path", default=".xanther/bench_neo4j_llm.db")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    if not args.api_key:
        print("ERROR: --api-key or OPENROUTER_API_KEY required"); raise SystemExit(1)
    r = asyncio.run(ingest_with_graph(
        data_path=args.data, project_id=args.project_id, api_key=args.api_key, model=args.model,
        user_id=args.user_id, limit=args.limit, concurrency=args.concurrency,
        neo4j_uri=args.neo4j_uri, neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password, sqlite_path=args.sqlite_path,
        resume=not args.no_resume,
    ))
    print(json.dumps({"questions": r["questions"]}, indent=2))
