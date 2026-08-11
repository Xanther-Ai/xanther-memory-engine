"""
LongMemEval Neo4j Retrieval v6 — v5 + "today" date injection + higher token budget.

Fixes for Zep (63.8%) gap:
1. Temporal: inject "Today's date is X" derived from max(haystack_dates) into prompt
2. Counting: max_tokens=600 so listing all items doesn't truncate
3. Keep type-aware system prompts from v5
"""
from __future__ import annotations
import argparse, asyncio, json, logging, os, re, time
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

TWO TYPES — identify which applies:

TYPE A: "How many days/weeks/months AGO" or "how long since I did X" → compare event date to today's date.
TYPE B: "Between [event 1] and [event 2]" or "order of events" → find BOTH event dates and compare them.

Steps:
1. Identify the type (A or B).
2. For TYPE A: Find the event date in "ALL FACTS BY DATE". Subtract from today's date.
3. For TYPE B: Find BOTH event dates from the facts/transcripts. Compute the difference between the two dates.
   IMPORTANT: Do NOT use today's date for type B unless explicitly asked.
4. Show calculation: "Event X = [date]. Event Y = [date]. Difference = N."
5. Convert: 7 days = 1 week, 30 days ≈ 1 month.

Only say "I don't know" if neither event date appears anywhere in the context."""

_SYSTEM_COUNTING = _SYSTEM_BASE + """

This is a COUNTING or AGGREGATION question.
1. Scan EVERYTHING in the context — facts AND conversation excerpts.
2. List each distinct item you find (do NOT stop early).
3. State the final answer as: "The answer is N." or "You have N [items]." or "The total is $X."
Always end with a clear final answer sentence."""

_SYSTEM_ORDERING = _SYSTEM_BASE + """

This is an ORDERING/SEQUENCE question.
Steps:
1. Find the date of each named event in the context.
2. Sort them chronologically.
3. State the order from first to last."""

_SYSTEM_PREFERENCE = _SYSTEM_BASE + """

This is a PREFERENCE question. The answer should DESCRIBE the user's preferences, not give generic recommendations.

Steps:
1. Find what specific tools, brands, topics, or activities the user uses, owns, or has shown interest in.
2. Describe their preference: "The user would prefer X because they use/have/mentioned Y."
3. Be specific about what they already use — that IS their preference.

Example: User has been asking about Adobe Premiere Pro → "The user would prefer resources for Adobe Premiere Pro specifically, as they are already using it for video editing." """

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
        return _SYSTEM_TEMPORAL, 500
    if "temporal" in qt and not is_elapsed and not is_ordering:
        # temporal type but counting-style (e.g. "how many days did I spend X") → counting
        return _SYSTEM_COUNTING, 600
    if is_ordering:
        return _SYSTEM_ORDERING, 400
    if is_elapsed:
        return _SYSTEM_TEMPORAL, 500
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


def _extract_key_terms(question: str) -> str:
    """Extract the most distinctive terms from a question for a targeted FTS pass.
    Focuses on proper nouns, place names, brand names — high-specificity terms
    that are most likely to appear in the relevant session."""
    # Remove common question prefixes
    q = re.sub(r"(?i)^(can you remind me|i remember|i was wondering|what was|what is|tell me about|"
               r"remind me of|going back to|checking our|i'm checking|i'm planning|"
               r"what did you|you told me|you mentioned|you recommended)[^,]*,?\s*", "", question)
    # Keep words with capital letters (proper nouns) and long words
    words = q.split()
    key = []
    for w in words:
        clean = re.sub(r"[^a-zA-Z0-9]", "", w)
        if not clean:
            continue
        # Proper noun (capitalized, 3+ chars) or long word (7+ chars)
        if (clean[0].isupper() and len(clean) >= 3) or len(clean) >= 7:
            if clean.lower() not in {"remind", "wondering", "previous", "conversation",
                                      "mentioned", "recommended", "planning", "remember",
                                      "checking", "revisit", "telling", "another"}:
                key.append(clean)
    return " ".join(key[:5]) if key else ""


def _dedup_facts(facts: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for f in sorted(facts, key=lambda x: x.get("sdate") or "", reverse=True):
        key = (f.get("attr", ""), str(f.get("val", "")).lower().strip()[:80])
        if key not in seen:
            seen[key] = f
    return list(seen.values())


def _format_facts_full(facts: list[dict], max_chars: int = 5000, temporal_first: bool = False) -> str:
    if not facts:
        return ""

    # Group by attribute to detect updates — most recent value wins
    by_attr: dict[str, list] = {}
    for f in facts:
        attr = f.get("attr", "")
        by_attr.setdefault(attr, []).append(f)

    # Build "LATEST STATE" section (most recent value per attribute)
    latest_lines = ["LATEST KNOWN STATE (most recent value per attribute):"]
    for attr, entries in sorted(by_attr.items()):
        sorted_entries = sorted(entries, key=lambda x: x.get("sdate") or "", reverse=True)
        best = sorted_entries[0]
        date_str = f" [{best.get('sdate')}]" if best.get("sdate") else ""
        if len(sorted_entries) > 1:
            latest_lines.append(f"  - {attr}: {best.get('val')}{date_str}  ← UPDATED")
        else:
            latest_lines.append(f"  - {attr}: {best.get('val')}{date_str}")

    # Also build chronological section for temporal/counting questions
    by_date: dict[str, list] = {}
    for f in facts:
        d = f.get("sdate") or "undated"
        by_date.setdefault(d, []).append(f)
    chron_lines = ["\nALL FACTS BY DATE (for counting/temporal questions):"]
    for date in sorted(by_date.keys()):
        chron_lines.append(f"\n[Session date: {date}]")
        for f in by_date[date]:
            chron_lines.append(f"  - {f.get('attr')}: {f.get('val')}")

    if temporal_first:
        result = "\n".join(chron_lines) + "\n\n" + "\n".join(latest_lines)
    else:
        result = "\n".join(latest_lines) + "\n" + "\n".join(chron_lines)
    return result[:max_chars]


async def retrieve_hybrid(
    engine,
    tfg,
    user_id: str,
    question: str,
    project_id: str,
    top_k: int = 15,
    question_type: str = "",
) -> tuple[str, str]:
    # 1. All facts for this project, deduped + date-sorted
    all_facts = await tfg.get_all_facts(user_id, project_id)
    deduped = _dedup_facts(all_facts)
    fact_ctx = _format_facts_full(deduped, max_chars=5000, temporal_first=("temporal" in question_type.lower())) if deduped else ""

    # 2. Episodic: BM25-ranked transcripts + key-term second pass
    results = await engine.search(
        query=question, project_id=project_id,
        user_id=user_id, layers=["episodic"], limit=top_k,
    )
    seen_ids: set[str] = set()
    all_episodes = list(results.episodic)
    for r in all_episodes:
        seen_ids.add(r.item_id)

    # Second pass: key proper nouns / named entities for better recall
    key_terms = _extract_key_terms(question)
    if key_terms and key_terms.lower() != question.lower()[:len(key_terms)]:
        results2 = await engine.search(
            query=key_terms, project_id=project_id,
            user_id=user_id, layers=["episodic"], limit=top_k,
        )
        for r in results2.episodic:
            if r.item_id not in seen_ids:
                seen_ids.add(r.item_id)
                all_episodes.append(r)

    # Sort by score, take best 15
    all_episodes.sort(key=lambda r: r.score, reverse=True)
    # For assistant questions: keep full transcripts (table/list answers need full content)
    max_tx_per_session = 4000 if "assistant" in question_type.lower() else 2000
    max_sessions = 8 if "assistant" in question_type.lower() else 15
    max_epi_chars = 20000 if "assistant" in question_type.lower() else 16000
    parts = []
    for r in all_episodes[:max_sessions]:
        tx = r.data.get("full_transcript", "")[:max_tx_per_session]
        if tx:
            parts.append(tx)
    epi_ctx = "\n\n---\n\n".join(parts)[:max_epi_chars]

    if not fact_ctx and not epi_ctx:
        return "No relevant context found.", "none"

    blocks = []
    if "assistant" in question_type.lower():
        # For assistant questions: skip user facts (they don't help), show only transcripts
        # Give more space to transcripts so the answer isn't buried
        if epi_ctx:
            blocks.append("RELEVANT CONVERSATION EXCERPTS (look for what the ASSISTANT said):\n" + epi_ctx)
        src = "episodic"
    else:
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
                            question_type=q_type,
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
