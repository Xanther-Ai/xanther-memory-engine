"""
LongMemEval Neo4j Retrieval v6 — v5 + "today" date injection + higher token budget.

Fixes for Zep (63.8%) gap:
1. Temporal: inject "Today's date is X" derived from max(haystack_dates) into prompt
2. Counting: max_tokens=600 so listing all items doesn't truncate
3. Keep type-aware system prompts from v5
"""
from __future__ import annotations
import argparse, asyncio, json, logging, os, time
from pathlib import Path
from typing import Optional
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------- System prompts ----------

_SYSTEM_BASE = """You answer questions about a user from their personal memory context.
Use ONLY the information provided. Never use outside knowledge."""

_SYSTEM_TEMPORAL = _SYSTEM_BASE + """

This is a TEMPORAL REASONING question. You will be told today's date.
Steps:
1. Identify the relevant event dates from the context (check both KNOWN FACTS dates and CONVERSATION EXCERPTS session dates).
2. Compute the EXACT difference using today's date or the two event dates.
3. Show your arithmetic explicitly: "Date A = X, Date B = Y, difference = Z days/weeks/months."
4. Give the final number as your answer.
Convert months as 30 days each. Weeks as 7 days. Round to nearest whole number.
Only say "I don't know" if neither relevant date appears anywhere in the context."""

_SYSTEM_COUNTING = _SYSTEM_BASE + """

This is a COUNTING or AGGREGATION question.
Steps:
1. List EVERY distinct item of the requested type found anywhere in context (facts AND transcripts).
2. Do NOT stop after 2-3 — scan the entire context.
3. Count them (or sum their values for aggregation).
4. State: "I found N items: [full list]" then give the final answer."""

_SYSTEM_ORDERING = _SYSTEM_BASE + """

This is an ORDERING/SEQUENCE question.
Steps:
1. Find the date of each named event in the context.
2. Sort them chronologically.
3. State the order from first to last."""

_SYSTEM_PREFERENCE = _SYSTEM_BASE + """

This is a PREFERENCE/RECOMMENDATION question.
Steps:
1. Find what specific tools, products, services, or topics the user has used, owns, or has asked about in detail.
2. Their implicit preference = what they already use, research, or engage with.
3. Recommend based on THAT, not generic knowledge.
Example: user asks about Adobe Premiere Pro settings → recommend Premiere Pro resources."""

_SYSTEM_DEFAULT = _SYSTEM_BASE + """

Find the specific fact in the context and state it directly in one sentence.
Only say "I don't know" if the fact is completely absent."""

_PROMPT = """TODAY'S DATE: {today}

MEMORY CONTEXT:
{context}

QUESTION: {question}

Think step by step, then give your final answer in one sentence:"""

_PROMPT_NO_DATE = """MEMORY CONTEXT:
{context}

QUESTION: {question}

Think step by step, then give your final answer in one sentence:"""

# ---------- Helpers ----------

def _pick_system(question: str, question_type: str) -> tuple[str, int]:
    """Returns (system_prompt, max_tokens)."""
    qt = question_type.lower()
    q = question.lower()
    # Temporal: only when asking about elapsed time or ordering events by date
    # NOT when just counting items that happen to span days (e.g. "how many days did I spend camping")
    is_elapsed = any(w in q for w in [
        "how long ago", "passed since", "passed between",
        "weeks ago", "days ago", "months ago",
        "how many weeks ago", "how many days ago", "how many months ago",
    ])
    is_ordering = ("order" in q or "first" in q or "before" in q) and any(
        w in q for w in ["first", "last", "before", "after", "earliest", "latest"]
    )
    if "temporal" in qt and (is_elapsed or is_ordering):
        return _SYSTEM_TEMPORAL, 400
    if "temporal" in qt and not is_elapsed and not is_ordering:
        # temporal type but counting-style (e.g. "how many days did I spend X") → counting
        return _SYSTEM_COUNTING, 600
    if is_ordering:
        return _SYSTEM_ORDERING, 400
    if is_elapsed:
        return _SYSTEM_TEMPORAL, 400
    if "preference" in qt:
        return _SYSTEM_PREFERENCE, 300
    if any(w in q for w in ["how many", "how much", "total", "combined", "altogether", "count"]):
        return _SYSTEM_COUNTING, 600
    return _SYSTEM_DEFAULT, 300


def _get_today(q: dict) -> str:
    """Use question_date (canonical LongMemEval reference date = when question is asked).
    Falls back to day after last haystack session if not present."""
    qdate = q.get("question_date", "")
    if qdate:
        return qdate.split(" ")[0].replace("/", "-")
    # fallback: last haystack session date
    dates = q.get("haystack_dates", [])
    if dates:
        cleaned = [d.split(" ")[0].replace("/", "-") for d in dates if d]
        if cleaned:
            return max(cleaned)
    return ""


def _dedup_facts(facts: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for f in sorted(facts, key=lambda x: x.get("sdate") or "", reverse=True):
        key = (f.get("attr", ""), str(f.get("val", "")).lower().strip()[:80])
        if key not in seen:
            seen[key] = f
    return list(seen.values())


def _format_facts_full(facts: list[dict], max_chars: int = 5000) -> str:
    if not facts:
        return ""
    by_date: dict[str, list] = {}
    for f in facts:
        d = f.get("sdate") or "undated"
        by_date.setdefault(d, []).append(f)
    lines = ["KNOWN FACTS ABOUT THE USER (all sessions, chronological):"]
    for date in sorted(by_date.keys()):
        lines.append(f"\n[Session date: {date}]")
        for f in by_date[date]:
            lines.append(f"  - {f.get('attr')}: {f.get('val')}")
    return "\n".join(lines)[:max_chars]


async def retrieve_hybrid(
    engine,
    tfg,
    user_id: str,
    question: str,
    project_id: str,
    top_k: int = 15,
) -> tuple[str, str]:
    # 1. All facts for this project, deduped + date-sorted
    all_facts = await tfg.get_all_facts(user_id, project_id)
    deduped = _dedup_facts(all_facts)
    fact_ctx = _format_facts_full(deduped, max_chars=5000) if deduped else ""

    # 2. Episodic: BM25-ranked transcripts
    results = await engine.search(
        query=question, project_id=project_id,
        user_id=user_id, layers=["episodic"], limit=top_k,
    )
    parts = []
    for r in results.episodic[:12]:
        tx = r.data.get("full_transcript", "")[:2200]
        if tx:
            parts.append(tx)
    epi_ctx = "\n\n---\n\n".join(parts)[:15000]

    if not fact_ctx and not epi_ctx:
        return "No relevant context found.", "none"

    blocks = []
    if fact_ctx:
        blocks.append(fact_ctx)
    if epi_ctx:
        blocks.append("RELEVANT CONVERSATION EXCERPTS:\n" + epi_ctx)
    src = "hybrid" if (fact_ctx and epi_ctx) else ("graph" if fact_ctx else "episodic")
    return "\n\n".join(blocks)[:22000], src


async def run_neo4j_retrieval(
    data_path: str,
    project_id: str,
    api_key: str,
    model: str,
    output_path: str,
    user_id: str = "benchmark",
    limit: Optional[int] = None,
    top_k: int = 15,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "",
    sqlite_path: str = ".xanther/bench_neo4j_llm.db",
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

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        async with MemoryEngine(settings) as engine:
            with open(output_path, "a") as fout:
                for i, q in enumerate(questions):
                    q_id = q["question_id"]
                    if q_id in done_ids:
                        continue
                    q_project = f"{project_id}_{q_id}"
                    q_type = q.get("question_type", "")
                    today = _get_today(q)
                    try:
                        ctx, src = await retrieve_hybrid(
                            engine, tfg, user_id,
                            q["question"], q_project, top_k,
                        )
                        system, max_tok = _pick_system(q["question"], q_type)
                        pred = await call_llm(
                            q["question"], ctx, api_key, model, http_client,
                            system, max_tok, today,
                        )
                        result = {
                            "question_id": q_id,
                            "question_type": q_type,
                            "question": q["question"],
                            "gold_answer": q["answer"],
                            "prediction": pred,
                            "context_retrieved": ctx[:400],
                            "retrieval_source": src,
                            "today_used": today,
                        }
                        fout.write(json.dumps(result) + "\n")
                        fout.flush()
                        done_ids.add(q_id)
                        if len(done_ids) % 10 == 0:
                            elapsed = time.time() - start
                            logger.info("[%d/%d] %s src=%s today=%s — %.0fs",
                                        len(done_ids), len(questions),
                                        q_type, src, today, elapsed)
                    except Exception as e:
                        logger.warning("Error on %s: %s", q_id, e)
                        errors += 1
                        if errors > 5:
                            break

    await driver.close()
    return {"processed": len(done_ids), "errors": errors, "output": output_path}


async def call_llm(question, context, api_key, model, client, system, max_tokens, today,
                   base_url="https://openrouter.ai/api/v1"):
    if today:
        prompt = _PROMPT.format(context=context, question=question, today=today)
    else:
        prompt = _PROMPT_NO_DATE.format(context=context, question=question)
    r = await client.post(
        f"{base_url}/chat/completions",
        json={"model": model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ], "temperature": 0.0, "max_tokens": max_tokens},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--project-id", default="lme-neo4j-llm")
    p.add_argument("--user-id", default="benchmark")
    p.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--output", default="benchmarks/longmemeval/results/neo4j_llm/predictions.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    p.add_argument("--sqlite-path", default=".xanther/bench_neo4j_llm.db")
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
