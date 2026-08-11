"""
LongMemEval Phase 3: Evaluation
Scores predictions against gold answers using:
  1. Exact/substring match (fast, free, ~85% correlation with LLM judge)
  2. LLM-as-judge (accurate, requires API key)

Usage:
    # Fast scoring (string match):
    python -m benchmarks.longmemeval.evaluate \
        --predictions benchmarks/longmemeval/results/predictions.jsonl \
        --mode string

    # LLM judge (recommended for final numbers):
    python -m benchmarks.longmemeval.evaluate \
        --predictions benchmarks/longmemeval/results/predictions.jsonl \
        --mode llm \
        --api-key sk-or-... \
        --judge-model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = """\
You are an evaluation judge for a memory benchmark.
Given a question, gold answer, and a predicted answer, determine if the prediction is correct.
The prediction is correct if it contains the key information from the gold answer,
even if worded differently. Minor variations in phrasing are OK.
The prediction is WRONG if it says "I don't know", gives a different answer, or contradicts the gold.

Respond with ONLY: "correct" or "incorrect"
"""

_JUDGE_PROMPT = """\
Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}

Is the prediction correct? Answer with ONLY "correct" or "incorrect":"""


# ---------------------------------------------------------------------------
# String match scorer (fast, no API)
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def string_match(prediction: str, gold: str) -> bool:
    """True if gold answer appears in prediction (case-insensitive, normalized)."""
    pred_n = normalize(prediction)
    gold_n = normalize(gold)

    # Exact match
    if pred_n == gold_n:
        return True

    # Gold is substring of prediction
    if gold_n in pred_n:
        return True

    # All significant words of gold appear in prediction
    gold_words = [w for w in gold_n.split() if len(w) > 2]
    if gold_words and all(w in pred_n for w in gold_words):
        return True

    # Check if it's a "don't know" response
    dnk = ["i don't know", "i do not know", "no context", "not found", "not available",
           "cannot find", "no relevant", "unclear"]
    if any(d in pred_n for d in dnk):
        return False

    return False


# ---------------------------------------------------------------------------
# LLM judge scorer
# ---------------------------------------------------------------------------

async def llm_judge(
    question: str,
    gold: str,
    prediction: str,
    api_key: str,
    model: str,
    base_url: str,
) -> bool:
    prompt = _JUDGE_PROMPT.format(question=question, gold=gold, prediction=prediction)
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 5,
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        verdict = resp.json()["choices"][0]["message"]["content"].strip().lower()
        return verdict.startswith("correct") and "incorrect" not in verdict


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

async def evaluate(
    predictions_path: str,
    mode: str = "string",
    api_key: str = "",
    judge_model: str = "openai/gpt-4o-mini",
    base_url: str = "https://openrouter.ai/api/v1",
    output_path: str | None = None,
) -> dict:
    predictions = []
    for line in Path(predictions_path).read_text().splitlines():
        if line.strip():
            predictions.append(json.loads(line))

    logger.info("Evaluating %d predictions (mode=%s)", len(predictions), mode)

    by_type: dict[str, list[bool]] = defaultdict(list)
    scored = []
    correct_total = 0

    for i, pred in enumerate(predictions):
        q = pred["question"]
        gold = pred["gold_answer"]
        prediction = pred["prediction"]
        q_type = pred.get("question_type", "unknown")

        if mode == "string":
            is_correct = string_match(prediction, gold)
        else:
            try:
                is_correct = await llm_judge(q, gold, prediction, api_key, judge_model, base_url)
            except Exception as e:
                logger.warning("Judge error on %s: %s — falling back to string match", pred.get("question_id"), e)
                is_correct = string_match(prediction, gold)

        by_type[q_type].append(is_correct)
        correct_total += int(is_correct)
        scored.append({**pred, "correct": is_correct})

        if (i + 1) % 50 == 0:
            running_acc = correct_total / (i + 1) * 100
            logger.info("[%d/%d] running accuracy: %.1f%%", i + 1, len(predictions), running_acc)

    # Compute final scores
    total = len(predictions)
    overall = correct_total / total * 100 if total else 0

    by_type_scores = {
        q_type: {
            "correct": sum(v),
            "total": len(v),
            "accuracy": round(sum(v) / len(v) * 100, 1) if v else 0,
        }
        for q_type, v in sorted(by_type.items())
    }

    results = {
        "overall_accuracy": round(overall, 1),
        "correct": correct_total,
        "total": total,
        "mode": mode,
        "judge_model": judge_model if mode == "llm" else "string_match",
        "by_type": by_type_scores,
    }

    # Print table
    print("\n" + "=" * 60)
    print("XME LongMemEval Results")
    print("=" * 60)
    print(f"  Overall accuracy: {overall:.1f}%  ({correct_total}/{total})")
    print(f"  Scoring mode: {mode}" + (f" ({judge_model})" if mode == "llm" else ""))
    print()
    print("  By question type:")
    for qt, s in by_type_scores.items():
        bar = "█" * int(s["accuracy"] / 5)
        print(f"    {qt:<30} {s['accuracy']:5.1f}%  {bar}")
    print("=" * 60)

    # Comparison table
    print("\n  vs. other memory systems (LongMemEval-S, string-match / LLM judge):")
    comparisons = [
        ("XME v0.1 (this run)", overall, mode),
        ("Zep (Graphiti)",       63.8,   "LLM judge"),
        ("Mem0 (2025 paper)",    49.0,   "LLM judge"),
        ("MemPalace",            96.6,   "R@5 recall (different metric)"),
        ("Letta",                83.2,   "LLM judge"),
    ]
    for name, score, note in comparisons:
        marker = " ◄ YOU" if "XME" in name else ""
        print(f"    {name:<30} {score:5.1f}%   [{note}]{marker}")
    print()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(results, indent=2))
        logger.info("Results saved to %s", output_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Score LongMemEval predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--mode", choices=["string", "llm"], default="string",
                        help="string = fast/free, llm = accurate (needs API key)")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    if args.mode == "llm" and not args.api_key:
        print("ERROR: --api-key required for llm mode", file=__import__("sys").stderr)
        raise SystemExit(1)

    asyncio.run(evaluate(
        predictions_path=args.predictions,
        mode=args.mode,
        api_key=args.api_key,
        judge_model=args.judge_model,
        base_url=args.base_url,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
