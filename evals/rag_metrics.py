"""RAG quality metrics for the golden set, filling one results dict.

Per case (the same retrieval-independent-case filter as evals.recall):

- recall_at_3:     relevant reference chunks found among the first 3
                   retrieved results / total relevant reference chunks
                   (term-based ground truth, corpus denominators).
- precision_at_3:  relevant chunks in the top 3 / 3 retrieved slots.
- hit_rate_at_3:   1.0 when >=1 relevant chunk is in the top 3, else 0.0.
- mrr:             reciprocal rank of the first relevant chunk.
- faithfulness:    share of answer claims the retrieved evidence supports,
                   judged by the configured LLM chain (app.agents.llm);
                   lexical-overlap proxy when no judge is available.
- answer_relevance: 0-1 score of how directly the answer addresses the
                   question, same judge / same proxy fallback.

Answers come from the last functional run (evals/report.json), so judging
does not re-drive the chat pipeline. Retrieval is measured through the exact
serving path (embedder -> rank_fragments), which is keyword-fallback when
sentence-transformers is not installed.

Usage:
    .venv/bin/python -m evals.rag_metrics              # writes evals/rag_metrics_report.json
    .venv/bin/python -m evals.rag_metrics --out p.json

Cases that never enter the retrieval path (escalation, out-of-scope) and the
tool-answered case emit nulls in the lists; aggregates skip nulls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from app.agents.llm import LLM as LLMClient
from app.agents.llm import LLMError
from evals.recall import SKIPPED_IDS, case_ground_truth, chunk_key
from evals.run import load_dataset
from ingest.embedder import Embedder
from ingest.store import get_store, rank_fragments

K = 3
REPORT_PATH = Path(__file__).with_name("rag_metrics_report.json")
JUDGE_CONCURRENCY = 4
JUDGE_MAX_TOKENS = 1024

_METRIC_KEYS = (
    "recall_at_3",
    "precision_at_3",
    "hit_rate_at_3",
    "mrr",
    "faithfulness",
    "answer_relevance",
)

_JUDGE_SYSTEM = (
    "You are a precise, neutral evaluation judge for an airline support "
    "chatbot. Respond with ONLY a JSON object, no prose, no markdown fences."
)

_JUDGE_TEMPLATE = """
Question:
{question}

Retrieved evidence chunks (the only evidence the bot could ground on):
{contexts}

Bot answer:
{answer}

Tasks:
1. Split the answer into its individual factual claims. Ignore small talk,
   polite filler, and next-step offers.
2. For each claim, decide whether the retrieved evidence supports it.
3. Rate answer_relevance on a 1-5 integer scale: 5 = directly answers the
   question with the right substance; 3 = partially responsive or padding
   around a correct core; 1 = unrelated to the question.

Return exactly:
{{"claims": [{{"claim": "<claim text>", "supported": true|false}}],
  "answer_relevance": <1-5 integer>}}
"""


def _context_block(chunks: list) -> str:
    if not chunks:
        return "(no evidence retrieved)"
    numbered = []
    for i, chunk in enumerate(chunks[:K], 1):
        title = chunk.title or chunk.source_url or f"source {i}"
        numbered.append(f"[{i}] {title}\n{chunk.content}")
    return "\n\n".join(numbered)


def _parse_judge_json(text: str) -> dict | None:
    """Find the judge's JSON object even with fences or preamble."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _lexicon_overlap(answer: str, chunks: list) -> float:
    """No-judge fallback: share of content words that appear in the evidence."""
    words = set(re.findall(r"[a-z0-9]{3,}", (answer or "").lower()))
    if not words:
        return 0.0
    haystack = " ".join(chunk.content.lower() for chunk in chunks[:K])
    covered = sum(1 for w in words if w in haystack)
    return covered / len(words)


def _question_overlap_answer(question: str, answer: str) -> float:
    """No-judge fallback: question-term coverage weighted by answer length."""
    q_words = {
        w for w in re.findall(r"[a-z0-9]{3,}", question.lower())
        if w not in {"what", "how", "can", "the", "asked", "does"}
    }
    if not q_words:
        return 0.0
    a_words = re.findall(r"[a-z0-9]{3,}", (answer or "").lower())
    if not a_words:
        return 0.0
    hit = sum(1 for w in a_words if w in q_words)
    return min(1.0, hit / max(1, len(a_words)) * 2.0)  # 50% topical words = full credit


async def judge_case(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    case: dict,
    answer: str,
    chunks: list,
) -> tuple[float | None, float | None, bool]:
    """Returns (faithfulness, answer_relevance, judged). None on judge failure."""
    user_prompt = _JUDGE_TEMPLATE.format(
        question=case["question"],
        contexts=_context_block(chunks),
        answer=answer,
    )
    async with semaphore:
        try:
            text = await client.complete(
                [
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=JUDGE_MAX_TOKENS,
            )
        except (LLMError, RuntimeError):
            return None, None, False
    data = _parse_judge_json(text)
    if data is None:
        return None, None, False

    claims = data.get("claims") or []
    supported = [c for c in claims if c.get("supported") is True]
    faithfulness = len(supported) / len(claims) if claims else None

    try:
        relevance = float(data["answer_relevance"])
    except (KeyError, TypeError, ValueError):
        relevance = None
    else:
        relevance = max(0.0, min(relevance, 5.0)) / 5.0  # scale to 0-1
    return faithfulness, relevance, True


async def _active_answers(path: Path) -> dict[str, str]:
    """Case id -> answer, from the latest functional run's report."""
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        case["id"]: case.get("answer") or ""
        for case in report.get("cases", [])
        if case.get("answer")
    }


async def collect(answer_path: Path, k: int = K) -> dict:
    """Build per-case metrics and the aggregated results dict."""
    cases = load_dataset(Path(__file__).with_name("dataset.jsonl"))
    answers = await _active_answers(answer_path)

    embedder = Embedder()
    store = get_store()
    await store.ensure_schema()
    fragments = await store.dump_fragments()
    retrieval_mode = "vector" if embedder.available else "keyword-fallback"

    eval_cases = [
        case for case in cases
        if case["id"] not in SKIPPED_IDS and case_ground_truth(case).terms
    ]
    embeddings: list[list[float]] | None = None
    if embedder.available and eval_cases:
        vectors = await asyncio.to_thread(
            embedder.embed, [case["question"] for case in eval_cases]
        )
        embeddings = vectors

    judge = LLMClient()
    semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)
    judge_calls = 0

    cases_out = []
    for index, case in enumerate(eval_cases):
        embedding = embeddings[index] if embeddings else None
        ranked = rank_fragments(fragments, embedding, case["question"], k=k)
        truth = case_ground_truth(case)

        def relevant(chunk, _truth=truth) -> bool:
            return _truth.is_relevant(chunk_key(chunk), chunk.content)

        relevant_total = sum(1 for f in fragments if relevant(f))
        relevant_hits = sum(1 for c in ranked if relevant(c))
        first_rank = next((i for i, c in enumerate(ranked, 1) if relevant(c)), None)
        answer = answers.get(case["id"], "")

        faithfulness, relevance, judged = await judge_case(
            judge, semaphore, case, answer, ranked
        )
        judge_calls += 1 if judged else 0
        if faithfulness is None:
            faithfulness = _lexicon_overlap(answer, ranked)
        if relevance is None:
            relevance = _question_overlap_answer(case["question"], answer)

        cases_out.append(
            {
                "id": case["id"],
                "question": case["question"],
                "ground_truth": "adjudicated" if truth.ids is not None else "term-proxy",
                "total_relevant": relevant_total,
                "retrieved_in_top_k": relevant_hits,
                "relevant_ranks": [
                    i
                    for i, c in enumerate(ranked, 1)
                    if relevant(c)
                ],
                "first_relevant_rank": first_rank,
                "context_titles": [
                    c.title or c.source_url for c in ranked[:k]
                ],
                "recall_at_3": (
                    relevant_hits / relevant_total if relevant_total else None
                ),
                "precision_at_3": relevant_hits / k if k else None,
                "hit_rate_at_3": 1.0 if relevant_hits > 0 else 0.0,
                "mrr": (1.0 / first_rank) if first_rank else 0.0,
                "faithfulness": round(faithfulness, 4) if faithfulness is not None else None,
                "answer_relevance": round(relevance, 4) if relevance is not None else None,
                "judge_used": judged,
            }
        )

    results: dict[str, list] = {key: [] for key in _METRIC_KEYS}
    applicable_ids = [c["id"] for c in cases_out if c["total_relevant"] > 0]
    for case in eval_cases:
        applicable = next(c for c in cases_out if c["id"] == case["id"])
        if case["id"] not in applicable_ids:
            for key in _METRIC_KEYS:
                results[key].append(None)  # tool-answered: not measurable
            continue
        results["recall_at_3"].append(applicable["recall_at_3"])
        results["precision_at_3"].append(applicable["precision_at_3"])
        results["hit_rate_at_3"].append(applicable["hit_rate_at_3"])
        results["mrr"].append(applicable["mrr"])
        results["faithfulness"].append(applicable["faithfulness"])
        results["answer_relevance"].append(applicable["answer_relevance"])

    summary = {
        key: (
            round(
                sum(v for v in values if v is not None)
                / sum(1 for v in values if v is not None),
                4,
            )
            if any(v is not None for v in values)
            else None
        )
        for key, values in results.items()
    }
    judge_rate = (
        round(
            sum(1 for c in cases_out if c["judge_used"]) / len(cases_out), 4
        )
        if cases_out
        else None
    )
    ids_map = {
        case["id"]: {key: case[key] for key in _METRIC_KEYS if key in case}
        for case in cases_out
    }
    skipped_ids = [c["id"] for c in cases if c["id"] in SKIPPED_IDS]
    return {
        "description": (
            "RAG metrics over the golden set. Retrieval metrics use "
            "term-based ground truth; faithfulness/answer_relevance use an "
            "LLM judge on the latest functional-run answers, with a lexical "
            "proxy fallback. Nulls mark retrieval-independent or "
            "tool-answered cases."
        ),
        "retrieval_mode": retrieval_mode,
        "k": k,
        "judge_model_calls": judge_calls,
        "judge_coverage": judge_rate,
        "results": results,
        "summary": summary,
        "case_ids_map": ids_map,
        "skipped_ids": skipped_ids,
        "cases": cases_out,
    }


async def amain(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    report = await collect(Path(args.answers))
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== RAG metrics (k={report['k']}) ===")
    print(f"mode: {report['retrieval_mode']}  judge coverage: {report['judge_coverage']}")
    for key, value in report["summary"].items():
        n = sum(1 for v in report["results"][key] if v is not None)
        print(f"  {key}: mean={value}  (n={n})")
    print(f"report: {args.out}  ({time.perf_counter() - started:.1f}s)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill the RAG results dict.")
    parser.add_argument("--out", default=str(REPORT_PATH))
    parser.add_argument(
        "--answers",
        default=str(REPORT_PATH.parent / "report.json"),
        help="Functional-run report with bot answers (default evals/report.json).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(parse_args())))
