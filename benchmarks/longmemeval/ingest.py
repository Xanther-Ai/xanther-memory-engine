"""
LongMemEval Phase 1: Ingestion
Feeds LongMemEval sessions into XME via session_start/record_turn/session_end.
Each question has ~54 sessions (haystack). We ingest them all then query.

Usage:
    python -m benchmarks.longmemeval.ingest \
        --data benchmarks/longmemeval/data/raw/longmemeval_s \
        --project-id longmemeval-xme \
        --limit 50   # subset for fast iteration
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_benchmark(data_path: str) -> list[dict]:
    with open(data_path) as f:
        return json.load(f)


async def ingest_question(
    engine,
    question: dict,
    project_id: str,
    user_id: str = "benchmark",
) -> dict:
    """Ingest all haystack sessions for one benchmark question."""
    q_id = question["question_id"]
    sessions = question["haystack_sessions"]  # list of sessions, each is list of turns
    dates = question.get("haystack_dates", [])

    ingested = 0
    for i, session_turns in enumerate(sessions):
        if not session_turns:
            continue

        date_hint = dates[i] if i < len(dates) else ""
        session_id_str = question.get("haystack_session_ids", [])[i] if i < len(question.get("haystack_session_ids", [])) else f"s{i}"

        # Start session
        ctx = await engine.session_start(
            project_id=project_id,
            user_id=user_id,
            session_id=f"{q_id}_{session_id_str}",
        )

        # Record all turns
        for turn in session_turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            engine.record_turn(ctx.session_id, role, content[:1000])

        # End session with date metadata
        summary = f"Session {i+1}/{len(sessions)}" + (f" ({date_hint})" if date_hint else "")
        await engine.session_end(
            session_id=ctx.session_id,
            project_id=project_id,
            user_id=user_id,
            summary=summary,
            outcome="success",
        )
        ingested += 1

    return {"question_id": q_id, "sessions_ingested": ingested}


async def ingest_all(
    data_path: str,
    project_id: str,
    user_id: str = "benchmark",
    limit: int | None = None,
    fallback_mode: bool = True,
    resume: bool = True,
) -> dict:
    """Ingest all (or subset of) benchmark questions into XME."""
    from xme.config import XMESettings
    from xme.engine import MemoryEngine

    questions = load_benchmark(data_path)
    if limit:
        questions = questions[:limit]

    # Track progress
    progress_file = Path(data_path).parent / f"ingest_progress_{project_id}.json"
    done_ids: set[str] = set()
    if resume and progress_file.exists():
        done = json.loads(progress_file.read_text())
        done_ids = set(done.get("completed", []))
        logger.info("Resuming: %d already ingested", len(done_ids))

    settings = XMESettings(
        sqlite_path=f".xanther/bench_{project_id}.db",
        fallback_mode=fallback_mode,
        opensearch_enabled=not fallback_mode,
    )

    results = []
    start = time.time()

    async with MemoryEngine(settings) as engine:
        # Disable LLM extraction during bulk ingestion for speed.
        # We store episodes verbatim; retrieval uses semantic search on transcripts.
        engine._extractor._api_key = ""
        for i, q in enumerate(questions):
            q_id = q["question_id"]
            if q_id in done_ids:
                continue

            t0 = time.time()
            # Isolate each question: separate project_id per haystack
            q_project_id = f"{project_id}_{q['question_id']}"
            r = await ingest_question(engine, q, q_project_id, user_id)
            elapsed = time.time() - t0
            r["elapsed_s"] = round(elapsed, 2)
            results.append(r)
            done_ids.add(q_id)

            if (i + 1) % 10 == 0:
                total_elapsed = time.time() - start
                rate = (i + 1) / total_elapsed
                eta = (len(questions) - i - 1) / rate if rate > 0 else 0
                logger.info(
                    "[%d/%d] ingested — %.1fs/q — ETA %.0fm",
                    i + 1, len(questions), elapsed, eta / 60,
                )
                # Save progress
                progress_file.write_text(json.dumps({"completed": list(done_ids)}))

    progress_file.write_text(json.dumps({"completed": list(done_ids)}))
    logger.info("✓ Ingestion complete: %d questions in %.1fs", len(results), time.time() - start)
    return {"questions_ingested": len(results), "results": results}


def main():
    parser = argparse.ArgumentParser(description="Ingest LongMemEval into XME")
    parser.add_argument("--data", required=True, help="Path to longmemeval_s data file")
    parser.add_argument("--project-id", default="longmemeval-xme")
    parser.add_argument("--user-id", default="benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Only ingest first N questions")
    parser.add_argument("--full", action="store_true", help="Use Neo4j+OpenSearch (default: SQLite)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from previous run")
    args = parser.parse_args()

    result = asyncio.run(ingest_all(
        data_path=args.data,
        project_id=args.project_id,
        user_id=args.user_id,
        limit=args.limit,
        fallback_mode=not args.full,
        resume=not args.no_resume,
    ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
