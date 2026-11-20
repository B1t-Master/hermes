"""Merge the eval reports into one summary JSON.

Reads the three reports produced by the eval modules:

- ``evals/report.json``             (evals.run: functional pass/fail, latency)
- ``evals/recall_report.json``      (evals.recall: recall@k, MRR)
- ``evals/rag_metrics_report.json`` (evals.rag_metrics: RAG quality metrics)

and writes ``evals/summary.json``. The default is a compact aggregate summary
suitable for documentation; ``--full`` adds one row per golden case with every
metric available for it. Answer text and per-chunk retrieval scores stay in the
source reports.

Usage:
    .venv/bin/python -m evals.summary            # compact summary for docs
    .venv/bin/python -m evals.summary --full     # include one row per case
    .venv/bin/python -m evals.summary --out p.json

The functional report is required; recall and RAG metrics are optional and
degrade to ``null`` (with a warning) when their report is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = _HERE / "report.json"
DEFAULT_RECALL_PATH = _HERE / "recall_report.json"
DEFAULT_RAG_PATH = _HERE / "rag_metrics_report.json"
DEFAULT_OUT_PATH = _HERE / "summary.json"

_FUNCTIONAL_KEYS = (
    "passed",
    "escalated",
    "cited",
    "latency_ms",
    "failed_checks",
    "error",
)
_RECALL_KEYS = (
    "ground_truth",
    "total_relevant",
    "retrieved_in_top_k",
    "relevant_ranks",
    "hits",
    "first_relevant_rank",
)
_RAG_KEYS = (
    "recall_at_3",
    "precision_at_3",
    "hit_rate_at_3",
    "mrr",
    "faithfulness",
    "answer_relevance",
    "judge_used",
)
_COMPACT_RAG_KEYS = (
    "recall_at_3",
    "precision_at_3",
    "hit_rate_at_3",
    "mrr",
    "faithfulness",
    "answer_relevance",
)


def load_report(path: Path) -> dict | None:
    """Read one report, returning None when it is absent or unparseable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"warning: report not found: {path}")
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: cannot read {path}: {exc}")
    return None


def _display_path(path: Path | None) -> str | None:
    """Prefer a repo-relative path so the summary is portable."""
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(_HERE.parent))
    except ValueError:
        return str(path)


def _cases_by_id(report: dict | None) -> dict[str, dict]:
    if not report:
        return {}
    return {
        case["id"]: case
        for case in report.get("cases") or []
        if isinstance(case, dict) and case.get("id")
    }


def _pick(case: dict | None, keys: tuple[str, ...]) -> dict | None:
    if case is None:
        return None
    return {key: case.get(key) for key in keys}


def _overall(report: dict | None, recall: dict | None, rag: dict | None) -> dict:
    passed = report.get("passed") if report else None
    total = report.get("total") if report else None
    overall = {
        "passed": passed,
        "total": total,
        "pass_rate": (
            round(passed / total, 4)
            if passed is not None and total
            else None
        ),
        "mean_latency_ms": report.get("mean_latency_ms") if report else None,
        "wall_clock_ms": report.get("wall_clock_ms") if report else None,
        "retrieval": None,
        "rag": None,
    }
    if recall:
        overall["retrieval"] = {
            "mode": recall.get("mode"),
            "corpus_chunks": recall.get("corpus_chunks"),
            "cases_scored": recall.get("cases_scored"),
            "server_retriever_k": recall.get("server_retriever_k"),
            "recall_by_k": recall.get("recall_by_k"),
            "case_hit_rate_at_server_k": recall.get("case_hit_rate_at_server_k"),
            "mrr_first_relevant": recall.get("mrr_first_relevant"),
        }
    if rag:
        overall["rag"] = {
            "k": rag.get("k"),
            "retrieval_mode": rag.get("retrieval_mode"),
            "judge_model_calls": rag.get("judge_model_calls"),
            "judge_coverage": rag.get("judge_coverage"),
            "summary": rag.get("summary"),
        }
    return overall


def build_full_summary(
    reports: dict[str, dict | None],
    *,
    paths: dict[str, Path] | None = None,
) -> dict:
    """Merge functional, recall, and RAG reports into one detailed summary."""
    paths = paths or {}
    functional = reports.get("functional")
    recall = reports.get("recall")
    rag = reports.get("rag_metrics")

    functional_cases = _cases_by_id(functional)
    recall_cases = _cases_by_id(recall)
    rag_cases = _cases_by_id(rag)

    warnings: list[str] = []
    if functional is None:
        warnings.append("functional report is missing; no pass/fail data")
    if recall is None:
        warnings.append("recall report is missing; retrieval metrics are null")
    if rag is None:
        warnings.append("RAG metrics report is missing; RAG metrics are null")

    ordered_ids = list(functional_cases)
    seen = set(ordered_ids)
    for source, name in ((recall_cases, "recall"), (rag_cases, "rag_metrics")):
        for case_id in source:
            if case_id not in seen:
                seen.add(case_id)
                ordered_ids.append(case_id)
                if functional_cases:
                    warnings.append(f"{case_id} present in {name} but not in functional report")

    # Report a case count mismatch only when a case is absent for a reason the
    # report did not declare (e.g. recall's skipped escalation/out-of-scope ids).
    for source, name in ((recall, "recall"), (rag, "rag_metrics")):
        if not source:
            continue
        declared_absent = set(source.get("skipped_ids") or [])
        reported_ids = set(_cases_by_id(source))
        for case_id in functional_cases:
            if case_id not in reported_ids and case_id not in declared_absent:
                warnings.append(
                    f"{case_id} missing from {name} report and not declared skipped"
                )

    cases: list[dict] = []
    for case_id in ordered_ids:
        functional_case = functional_cases.get(case_id)
        recall_case = recall_cases.get(case_id)
        rag_case = rag_cases.get(case_id)
        question = None
        for case in (functional_case, recall_case, rag_case):
            if case and case.get("question"):
                question = case["question"]
                break

        missing = [
            name
            for name, case in (
                ("functional", functional_case),
                ("recall", recall_case),
                ("rag_metrics", rag_case),
            )
            if case is None
        ]
        cases.append(
            {
                "id": case_id,
                "question": question,
                "functional": _pick(functional_case, _FUNCTIONAL_KEYS),
                "recall": _pick(recall_case, _RECALL_KEYS),
                "rag": _pick(rag_case, _RAG_KEYS),
                "missing": missing,
            }
        )

    sources = {
        "functional": {
            "path": _display_path(paths.get("functional", DEFAULT_REPORT_PATH)),
            "present": functional is not None,
            "cases": len(functional_cases) if functional is not None else None,
        },
        "recall": {
            "path": _display_path(paths.get("recall", DEFAULT_RECALL_PATH)),
            "present": recall is not None,
            "cases": len(recall_cases) if recall is not None else None,
        },
        "rag_metrics": {
            "path": _display_path(paths.get("rag_metrics", DEFAULT_RAG_PATH)),
            "present": rag is not None,
            "cases": len(rag_cases) if rag is not None else None,
        },
    }

    return {
        "generated_at": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "sources": sources,
        "overall": _overall(functional, recall, rag),
        "cases": cases,
        "warnings": warnings,
    }


def compact_summary(full: dict) -> dict:
    """Drop per-case detail, keeping only aggregates for documentation."""
    overall = full["overall"]
    retrieval = overall.get("retrieval")
    rag = overall.get("rag")
    retrieval_k = retrieval.get("server_retriever_k") if retrieval else None

    compact = {
        "generated_at": full["generated_at"],
        "functional": {
            "passed": overall["passed"],
            "total": overall["total"],
            "pass_rate": overall["pass_rate"],
            "mean_latency_ms": overall["mean_latency_ms"],
        },
        "retrieval": (
            {
                "mode": retrieval.get("mode"),
                "corpus_chunks": retrieval.get("corpus_chunks"),
                "k": retrieval_k,
                "recall_at_k": (
                    retrieval.get("recall_by_k") or {}
                ).get(str(retrieval_k)),
                "case_hit_rate_at_k": retrieval.get("case_hit_rate_at_server_k"),
                "mrr": retrieval.get("mrr_first_relevant"),
            }
            if retrieval
            else None
        ),
        "rag": (
            {
                "k": rag.get("k"),
                **{
                    key: (rag.get("summary") or {}).get(key)
                    for key in _COMPACT_RAG_KEYS
                },
            }
            if rag
            else None
        ),
    }
    if full["warnings"]:
        compact["warnings"] = full["warnings"]
    return compact


def build_summary(
    reports: dict[str, dict | None],
    *,
    paths: dict[str, Path] | None = None,
    full: bool = False,
) -> dict:
    """Build a compact summary by default; ``full=True`` keeps per-case rows."""
    detail = build_full_summary(reports, paths=paths)
    return detail if full else compact_summary(detail)


def amain(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    recall_path = Path(args.recall)
    rag_path = Path(args.rag)

    reports = {
        "functional": load_report(report_path),
        "recall": load_report(recall_path),
        "rag_metrics": load_report(rag_path),
    }
    if reports["functional"] is None:
        print(f"error: functional report is required: {report_path}")
        return 1

    detail = build_full_summary(
        reports,
        paths={
            "functional": report_path,
            "recall": recall_path,
            "rag_metrics": rag_path,
        },
    )
    summary = detail if args.full else compact_summary(detail)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    overall = detail["overall"]
    print("=== eval summary ===")
    print(f"functional: {overall['passed']}/{overall['total']} passed "
          f"({(overall['pass_rate'] or 0) * 100:.1f}%)")
    retrieval = overall["retrieval"]
    if retrieval:
        k = retrieval["server_retriever_k"]
        recall_at_k = (retrieval.get("recall_by_k") or {}).get(str(k))
        print(f"retrieval: recall@{k}={recall_at_k}  mode={retrieval['mode']}")
    rag = overall["rag"]
    if rag:
        means = rag.get("summary") or {}
        print(f"rag: faithfulness={means.get('faithfulness')}  "
              f"answer_relevance={means.get('answer_relevance')}")
    print(f"cases: {len(detail['cases'])}  warnings: {len(detail['warnings'])}")
    print(f"rows: {'full per-case' if args.full else 'compact (use --full for per-case)'}")
    print(f"report: {out_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the eval reports into one summary JSON."
    )
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--recall", default=str(DEFAULT_RECALL_PATH))
    parser.add_argument("--rag", default=str(DEFAULT_RAG_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include one row per golden case instead of aggregates only.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(amain(parse_args()))
