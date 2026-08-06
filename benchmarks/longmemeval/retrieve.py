"""
LongMemEval Phase 2: Retrieval
For each benchmark question, retrieves relevant context from XME,
then calls an LLM to produce an answer.

Usage:
    python -m benchmarks.longmemeval.retrieve \
        --data benchmarks/longmemeval/data/raw/longmemeval_s \
        --project-id longmemeval-xme \
        --api-key sk-or-... \
        --model openai/gpt-4o-mini \
        --output benchmarks/longmemeval/results/predictions.jsonl \
        --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a helpful assistant with access to the user's conversation history.
Answer the question using ONLY the provided conversation context.
Be concise and factual. If the answer is not in the context, say "I don't know".
Do not make up information.
"""

_ANSWER_PROMPT = """\
Conversation history retrieved for this question:
{context}

---
Question: {question}

Answer in 1-3 sentences:"""


async def retrieve_context(
    engine,
    question: str,
    project_id: str,
    user_id: str,
    top_k: int = 10,
) -> str:
    """Search XME for relevant context, return formatted string."""
    results = await engine.search(
        query=question,
        project_id=project_id,
        user_id=user_id,
        layers=["episodic", "facts"],
        limit=top_k,
    )

    parts: list[str] = []
    # Episodic results — direct session content
    for r in results.episodic[:6]:
        d = r.data
        # Get transcript excerpt if available
        transcript = d.get("full_transcript", d.get("summary", ""))
        if transcript:
            parts.append(f"[Session] {transcript[:500]}")

    # Fact results — structured extracted knowledge
    for r in results.facts[:4]:
        d = r.data
        content = d.get("content", d.get("title", ""))
        if content:
            parts.append(f"[Fact] {content[:200]}")

    return "\n\n".join(parts) if parts else "No relevant context found."


async def call_llm(
    question: str,
    context: str,
    api_key: str,
    model: str,
    base_url: str = "https://openrouter.ai/api/v1",
) -> str:
    """Call LLM with retrieved context to generate answer."""
    prompt = _ANSWER_PROMPT.format(context=context, question=question)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def process_question(
    engine,
    question: dict,
    project_id: str,
    user_id: str,
    api_key: str,
    model: str,
    top_k: int = 10,
) -> dict:
    q_id = question["question_id"]
    q_text = question["question"]
    gold = question["answer"]
    q_type = question.get("question_type", "unknown")

    # Retrieve
    t0 = time.time()
    context = await retrieve_context(engine, q_text, project_id, user_id, top_k)
    retrieve_ms = int((time.time() - t0) * 1000)

    # Generate answer
    t1 = time.time()
    prediction = await call_llm(q_text, context, api_key, model)
    gen_ms = int((time.time() - t1) * 1000)

    return {
        "question_id": q_id,
        "question_type": q_type,
        "question": q_text,
        "gold_answer": gold,
        "prediction": prediction,
        "context_retrieved": context[:500],  # truncate for storage
        "retrieve_ms": retrieve_ms,
        "gen_ms": gen_ms,
    }


async def run_retrieval(
    data_path: str,
    project_id: str,
    api_key: str,
    model: str,
    output_path: str,
    user_id: str = "benchmark",
    limit: int | None = None,
    top_k: int = 10,
    fallback_mode: bool = True,
    resume: bool = True,
) -> dict:
    from xme.config import XMESettings
    from xme.engine import MemoryEngine

    questions = json.loads(Path(data_path).read_text())
    if limit:
        questions = questions[:limit]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load already-done question IDs
    done_ids: set[str] = set()
    if resume and output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["question_id"])
        logger.info("Resuming: %d predictions already done", len(done_ids))

    settings = XMESettings(
        sqlite_path=f".xanther/bench_{project_id}.db",
        fallback_mode=fallback_mode,
        opensearch_enabled=not fallback_mode,
    )

    total_correct = 0
    total_done = len(done_ids)
    start = time.time()
    errors = 0

    async with MemoryEngine(settings) as engine:
        with open(output, "a") as fout:
            for i, q in enumerate(questions):
                q_id = q["question_id"]
                if q_id in done_ids:
                    continue

                try:
                    result = await process_question(
                        engine, q, project_id, user_id, api_key, model, top_k
                    )
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    total_done += 1

                    if (total_done) % 10 == 0:
                        elapsed = time.time() - start
                        rate = total_done / elapsed if elapsed > 0 else 1
                        remaining = (len(questions) - total_done)
                        eta = remaining / rate
                        logger.info(
                            "[%d/%d] — %.0fs elapsed — ETA %.0fm",
                            total_done, len(questions), elapsed, eta / 60,
                        )
                except Exception as e:
                    logger.warning("Error on %s: %s", q_id, e)
                    errors += 1
                    if errors > 10:
                        logger.error("Too many errors, aborting")
                        break

    return {
        "total_processed": total_done,
        "output": str(output),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval retrieval + answer generation")
    parser.add_argument("--data", required=True)
    parser.add_argument("--project-id", default="longmemeval-xme")
    parser.add_argument("--user-id", default="benchmark")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--output", default="benchmarks/longmemeval/results/predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: --api-key or OPENROUTER_API_KEY required", file=__import__("sys").stderr)
        raise SystemExit(1)

    result = asyncio.run(run_retrieval(
        data_path=args.data,
        project_id=args.project_id,
        api_key=args.api_key,
        model=args.model,
        output_path=args.output,
        user_id=args.user_id,
        limit=args.limit,
        top_k=args.top_k,
        fallback_mode=not args.full,
        resume=not args.no_resume,
    ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
