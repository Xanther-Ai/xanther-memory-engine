"""
LongMemEval full pipeline runner.
Orchestrates ingest → retrieve → evaluate in sequence.

Usage (quick smoke test on 10 questions):
    python -m benchmarks.longmemeval.run \
        --data benchmarks/longmemeval/data/raw/longmemeval_s \
        --api-key sk-or-... \
        --limit 10 \
        --output-dir benchmarks/longmemeval/results/run_001

Usage (full 500-question run):
    python -m benchmarks.longmemeval.run \
        --data benchmarks/longmemeval/data/raw/longmemeval_s \
        --api-key sk-or-... \
        --output-dir benchmarks/longmemeval/results/run_001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_pipeline(
    data_path: str,
    api_key: str,
    output_dir: str,
    project_id: str = "longmemeval-xme",
    user_id: str = "benchmark",
    answer_model: str = "openai/gpt-4o-mini",
    judge_model: str = "openai/gpt-4o-mini",
    judge_mode: str = "string",
    limit: int | None = None,
    top_k: int = 10,
    fallback_mode: bool = True,
    skip_ingest: bool = False,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions_path = str(out / "predictions.jsonl")
    results_path = str(out / "results.json")

    total_start = time.time()

    # --- Phase 1: Ingest ---
    if not skip_ingest:
        logger.info("=" * 50)
        logger.info("PHASE 1: Ingesting sessions into XME")
        logger.info("=" * 50)
        from benchmarks.longmemeval.ingest import ingest_all
        ingest_result = await ingest_all(
            data_path=data_path,
            project_id=project_id,
            user_id=user_id,
            limit=limit,
            fallback_mode=fallback_mode,
            resume=True,
        )
        (out / "ingest_summary.json").write_text(json.dumps(ingest_result, indent=2))
        logger.info("✓ Phase 1 complete: %d questions ingested", ingest_result["questions_ingested"])
    else:
        logger.info("Skipping ingestion (--skip-ingest)")

    # --- Phase 2: Retrieve + Answer ---
    logger.info("=" * 50)
    logger.info("PHASE 2: Retrieving context + generating answers")
    logger.info("=" * 50)
    from benchmarks.longmemeval.retrieve import run_retrieval
    retrieve_result = await run_retrieval(
        data_path=data_path,
        project_id=project_id,
        api_key=api_key,
        model=answer_model,
        output_path=predictions_path,
        user_id=user_id,
        limit=limit,
        top_k=top_k,
        fallback_mode=fallback_mode,
        resume=True,
    )
    logger.info("✓ Phase 2 complete: %d predictions", retrieve_result["total_processed"])

    # --- Phase 3: Evaluate ---
    logger.info("=" * 50)
    logger.info("PHASE 3: Scoring predictions")
    logger.info("=" * 50)
    from benchmarks.longmemeval.evaluate import evaluate
    eval_result = await evaluate(
        predictions_path=predictions_path,
        mode=judge_mode,
        api_key=api_key if judge_mode == "llm" else "",
        judge_model=judge_model,
        output_path=results_path,
    )

    total_elapsed = time.time() - total_start
    summary = {
        "overall_accuracy": eval_result["overall_accuracy"],
        "total_questions": eval_result["total"],
        "correct": eval_result["correct"],
        "answer_model": answer_model,
        "judge_mode": judge_mode,
        "judge_model": judge_model,
        "top_k_retrieval": top_k,
        "total_elapsed_min": round(total_elapsed / 60, 1),
        "output_dir": str(out),
        "by_type": eval_result["by_type"],
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("=" * 50)
    logger.info("✓ COMPLETE: %.1f%% accuracy in %.1f min", summary["overall_accuracy"], summary["total_elapsed_min"])
    logger.info("Results: %s", results_path)
    logger.info("=" * 50)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run full LongMemEval pipeline on XME")
    parser.add_argument("--data", required=True, help="Path to longmemeval_s data file")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--output-dir", default="benchmarks/longmemeval/results/run_001")
    parser.add_argument("--project-id", default="longmemeval-xme")
    parser.add_argument("--user-id", default="benchmark")
    parser.add_argument("--answer-model", default="openai/gpt-4o-mini",
                        help="LLM for generating answers from retrieved context")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini",
                        help="LLM judge model (only used with --judge-mode llm)")
    parser.add_argument("--judge-mode", choices=["string", "llm"], default="string",
                        help="string = fast/free, llm = accurate")
    parser.add_argument("--limit", type=int, default=None, help="Run on first N questions only")
    parser.add_argument("--top-k", type=int, default=10, help="Number of memories to retrieve per question")
    parser.add_argument("--full", action="store_true", help="Use Neo4j+OpenSearch (default: SQLite fallback)")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion (use existing XME data)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: --api-key or OPENROUTER_API_KEY required", file=__import__("sys").stderr)
        raise SystemExit(1)

    result = asyncio.run(run_pipeline(
        data_path=args.data,
        api_key=args.api_key,
        output_dir=args.output_dir,
        project_id=args.project_id,
        user_id=args.user_id,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        judge_mode=args.judge_mode,
        limit=args.limit,
        top_k=args.top_k,
        fallback_mode=not args.full,
        skip_ingest=args.skip_ingest,
    ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
