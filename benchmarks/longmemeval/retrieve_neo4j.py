"""
LongMemEval Neo4j Retrieval — hybrid graph-first + episodic fallback.
Graph-first: query TemporalFactGraph for personal facts (fast, precise).
Episodic fallback: FTS search over session transcripts.
"""
from __future__ import annotations
import argparse, asyncio, json, logging, os, time
from pathlib import Path
from typing import Optional
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SYSTEM = """You are answering questions about a user based on their conversation history.
Use ONLY the provided memory context. Be direct and concise (1 sentence).
If the answer is in the context, state it directly.
If not found, say: I don't know."""

_PROMPT = """MEMORY CONTEXT:
{context}

QUESTION: {question}

Answer directly in one sentence:"""


async def retrieve_hybrid(
    engine,
    tfg,
    embedder,
    question: str,
    project_id: str,
    user_id: str,
    top_k: int = 10,
) -> tuple[str, str]:
    """Returns (context_str, retrieval_source)."""
    import sys; sys.path.insert(0, ".")
    from xme.layers.temporal_graph import extract_personal_facts

    # 1. Graph-first: query personal facts
    q_emb = embedder.embed(question)
    facts = await tfg.search_facts(question, user_id, project_id, embedding=q_emb, top_k=top_k)

    # Only trust graph if highest scoring fact is high confidence
    high_conf_facts = [f for f in facts if (f.get('score') or 0) > 0.70]
    if high_conf_facts:
        ctx = tfg.format_for_llm(high_conf_facts, max_chars=2000)
        return ctx, "graph"

    # 2. Episodic fallback: FTS + full transcript
    results = await engine.search(
        query=question, project_id=project_id,
        user_id=user_id, layers=["episodic"], limit=top_k,
    )
    if not results.episodic:
        return "No relevant context found.", "none"

    parts = []
    for r in results.episodic[:5]:
        tx = r.data.get("full_transcript", "")[:3000]
        if tx:
            parts.append(tx)
    return "\n\n---\n\n".join(parts)[:6000], "episodic"


async def run_neo4j_retrieval(
    data_path: str,
    project_id: str,
    api_key: str,
    model: str,
    output_path: str,
    user_id: str = "benchmark",
    limit: Optional[int] = None,
    top_k: int = 10,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "",
    sqlite_path: str = ".xanther/bench_neo4j.db",
    resume: bool = True,
) -> dict:
    import sys; sys.path.insert(0, ".")
    from xme.config import XMESettings
    from xme.engine import MemoryEngine
    from xme.layers.temporal_graph import TemporalFactGraph
    from xme.extraction.embedder import LocalEmbedder
    from neo4j import AsyncGraphDatabase

    questions = json.loads(Path(data_path).read_text())
    if limit:
        questions = questions[:limit]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    if resume and out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["question_id"])
        logger.info("Resuming: %d done", len(done_ids))

    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    tfg = TemporalFactGraph(driver)
    settings = XMESettings(sqlite_path=sqlite_path, fallback_mode=True, opensearch_enabled=False)
    embedder = LocalEmbedder()

    start = time.time()
    errors = 0

    async with MemoryEngine(settings) as engine:
        with open(output_path, "a") as fout:
            for i, q in enumerate(questions):
                q_id = q["question_id"]
                if q_id in done_ids:
                    continue
                q_project = f"{project_id}_{q_id}"
                try:
                    ctx, src = await retrieve_hybrid(
                        engine, tfg, embedder,
                        q["question"], q_project, user_id, top_k,
                    )
                    pred = await call_llm(q["question"], ctx, api_key, model)
                    result = {
                        "question_id": q_id,
                        "question_type": q.get("question_type",""),
                        "question": q["question"],
                        "gold_answer": q["answer"],
                        "prediction": pred,
                        "context_retrieved": ctx[:300],
                        "retrieval_source": src,
                    }
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    done_ids.add(q_id)
                    if (len(done_ids)) % 10 == 0:
                        elapsed = time.time() - start
                        logger.info("[%d/%d] src=%s — %.0fs elapsed", len(done_ids), len(questions), src, elapsed)
                except Exception as e:
                    logger.warning("Error on %s: %s", q_id, e)
                    errors += 1
                    if errors > 5:
                        break

    await driver.close()
    return {"processed": len(done_ids), "errors": errors, "output": output_path}


async def call_llm(question, context, api_key, model, base_url="https://openrouter.ai/api/v1"):
    prompt = _PROMPT.format(context=context, question=question)
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{base_url}/chat/completions",
            json={"model": model, "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ], "temperature": 0.0, "max_tokens": 150},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--project-id", default="lme-neo4j")
    p.add_argument("--user-id", default="benchmark")
    p.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY",""))
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--output", default="benchmarks/longmemeval/results/neo4j/predictions.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI","bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER","neo4j"))
    p.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD",""))
    p.add_argument("--sqlite-path", default=".xanther/bench_neo4j.db")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    if not args.api_key:
        print("ERROR: --api-key required"); raise SystemExit(1)
    r = asyncio.run(run_neo4j_retrieval(
        data_path=args.data, project_id=args.project_id,
        api_key=args.api_key, model=args.model, output_path=args.output,
        user_id=args.user_id, limit=args.limit, top_k=args.top_k,
        neo4j_uri=args.neo4j_uri, neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password, sqlite_path=args.sqlite_path,
        resume=not args.no_resume,
    ))
    print(json.dumps(r, indent=2))
