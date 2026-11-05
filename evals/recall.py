"""Recall@k for knowledge-base retrieval over the golden set.

Measures the table-defined metric directly: relevant reference chunks found
among the first k retrieved results / total relevant reference chunks.

Ground truth is deterministic, mirroring scoring.py's philosophy (no LLM
judge): a corpus chunk is *relevant* to a golden case when its content
contains one of the case's declared terms (``expect_any`` +
``expect_contains``, whitespace/dash-normalized). Relevance is judged over
the full corpus, so the denominator is every matching chunk, not just what
the top-k happened to return.

Ranking is the exact serving path: the golden question is embedded with the
serving embedder and ranked with :func:`ingest.store.rank_fragments` over the
dumped store index (keyword fallback when the embedder is unavailable), so a
recall number here is what the live pipeline would retrieve.

Cases without term expectations (e.g. the two escalation prompts) have no
retrieval ground truth and are dropped; ``out-of-scope`` is dropped too —
its terms describe the refusal wording, not knowledge-base content.

Usage:
    .venv/bin/python -m evals.recall            # writes evals/recall_report.json
    .venv/bin/python -m evals.recall --out path.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from app.config import settings
from evals.run import load_dataset
from evals.scoring import _normalize as normalize_term
from ingest.embedder import Embedder
from ingest.store import get_store, rank_fragments

DEFAULT_REPORT_PATH = Path(__file__).with_name("recall_report.json")
# No KB ground truth: escalation prompts never enter the retrieval path, and
# the out-of-scope question is answered by routing, not by chunks.
_SKIPPED_IDS = {"escalate-human", "escalate-safety", "out-of-scope"}
_MAX_K = 10

_DESCRIPTION = (
    "Recall@k over the golden set: relevant reference chunks found among the "
    "first k retrieved results / total relevant reference chunks. Relevance "
    "is term-based ground truth (chunk contains a declared case term), "
    "denominated over the full corpus."
)


def case_terms(case: dict) -> list[str]:
    """Normalized ground-truth terms for one case, deduped and sorted."""
    terms: list[str] = [normalize_term(t) for t in (case.get("expect_contains") or [])]
    terms += [normalize_term(t) for t in (case.get("expect_any") or [])]
    return sorted({t for t in terms if t})


def term_in_content(term: str, content: str) -> bool:
    """Word/phrase-boundary match so "no" doesn't hit inside "token".

    Term boundaries reflect how the keyword search scores query terms
    (``keyword_terms`` extracts \\w+ tokens; a term with spaces is matched as
    an exact phrase after punctuation-insensitive folding).
    """
    pattern = r"[\W]+".join(re.escape(part) for part in term.split())
    return re.search(rf"\b{pattern}(?!\w)", content) is not None


def is_relevant(terms: list[str], content: str) -> bool:
    """True when the chunk contains any of the case's ground-truth terms."""
    normalized = normalize_term(content)
    return any(term_in_content(term, normalized) for term in terms)


async def measure(max_k: int = _MAX_K) -> dict:
    """Score retrieval for every retrieval-dependent golden case."""
    cases = load_dataset(Path(__file__).with_name("dataset.jsonl"))
    embedder = Embedder()
    store = get_store()
    await store.ensure_schema()
    fragments = await store.dump_fragments()

    mode = "vector" if embedder.available else "keyword-fallback"
    query_embeddings: list[list[float]] | None = None
    eval_cases = [c for c in cases if c["id"] not in _SKIPPED_IDS and case_terms(c)]
    if embedder.available and eval_cases:
        vectors = await asyncio.to_thread(
            embedder.embed, [case["question"] for case in eval_cases]
        )
        query_embeddings = vectors

    per_case: list[dict] = []
    for index, case in enumerate(eval_cases):
        terms = case_terms(case)
        embedding = query_embeddings[index] if query_embeddings else None
        ranked = rank_fragments(fragments, embedding, case["question"], k=max_k)

        relevant_ranks = [
            i
            for i, chunk in enumerate(ranked, 1)
            if is_relevant(terms, chunk.content)
        ]
        corpus_count = sum(
            1 for fragment in fragments if is_relevant(terms, fragment.content)
        )

        hits_by_k = {
            str(k): sum(1 for rank in relevant_ranks if rank <= k)
            for k in range(1, max_k + 1)
        }
        per_case.append(
            {
                "id": case["id"],
                "question": case["question"],
                "terms": terms,
                "total_relevant": corpus_count,
                "retrieved_in_top_k": len(relevant_ranks),
                "relevant_ranks": relevant_ranks,
                "hits": hits_by_k,
                "first_relevant_rank": relevant_ranks[0] if relevant_ranks else None,
                "top_scores": [
                    {
                        "rank": i,
                        "title": chunk.title,
                        "score": round(chunk.score, 4),
                        "relevant": i in set(relevant_ranks),
                    }
                    for i, chunk in enumerate(ranked[:5], 1)
                ],
            }
        )

    # Corpus-level Recall@k per cut-off (metric denominator = every relevant
    # corpus chunk), plus the case hit-rate and MRR of the first relevant hit.
    server_k = settings.retriever_k
    denominator = sum(case["total_relevant"] for case in per_case)
    recall_by_k: dict[str, float | None] = {}
    for k in range(1, max_k + 1):
        key = str(k)
        recall_by_k[key] = (
            round(
                sum(case["hits"][key] for case in per_case) / denominator, 4
            ) if denominator else None
        )

    first_ranks = [
        case["first_relevant_rank"] for case in per_case if case["first_relevant_rank"]
    ]
    mrr = round(
        sum(1 / rank for rank in first_ranks) / len(first_ranks), 4
    ) if first_ranks else None
    # Case-level hit rate counts only cases where the corpus actually
    # contains term-matching chunks; cases whose answers come from tools
    # (no relevant chunks at all) are N/A, not misses.
    applicable = [case for case in per_case if case["total_relevant"] > 0]
    not_applicable_ids = [
        case["id"] for case in per_case if case["total_relevant"] == 0
    ]
    hit_rate = round(
        sum(1 for case in applicable if case["hits"].get(str(server_k), 0) > 0)
        / len(applicable),
        4,
    ) if applicable else None

    return {
        "description": _DESCRIPTION,
        "mode": mode,
        "corpus_chunks": len(fragments),
        "cases_scored": len(per_case),
        "cases_total": len(cases),
        "skipped_ids": sorted(_SKIPPED_IDS),
        "server_retriever_k": server_k,
        "recall_by_k": recall_by_k,
        "applicable_cases": len(applicable),
        "not_applicable_ids": not_applicable_ids,
        "case_hit_rate_at_server_k": hit_rate,
        "mrr_first_relevant": mrr,
        "cases": per_case,
    }


async def amain(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    report = await measure()
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== retrieval recall ===")
    print(f"mode: {report['mode']}  corpus chunks: {report['corpus_chunks']}  "
          f"cases: {report['cases_scored']}")
    print(f"  corpus-level recall@k (server k = {report['server_retriever_k']}):")
    for k, value in report["recall_by_k"].items():
        marker = "  <-- serving k" if int(k) == report["server_retriever_k"] else ""
        print(f"    k={k}: {value}{marker}")
    print(f"  case hit-rate@server_k: {report['case_hit_rate_at_server_k']}")
    print(f"  MRR (first relevant): {report['mrr_first_relevant']}")
    zero_cases = [
        f"{case['id']} ({case['retrieved_in_top_k']}/{case['total_relevant']})"
        for case in report["cases"]
        if case["total_relevant"] > 0 and case["retrieved_in_top_k"] == 0
    ]
    if zero_cases:
        print(f"  zero-hit cases: {', '.join(zero_cases)}")
    if report["not_applicable_ids"]:
        print(f"  not applicable (no relevant chunks in corpus): "
              f"{', '.join(report['not_applicable_ids'])}")
    print(f"report: {out_path}  ({time.perf_counter() - started:.1f}s)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score recall@k for KB retrieval.")
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH))
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(parse_args())))
