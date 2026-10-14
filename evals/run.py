"""Run the hermes golden-set evaluation against a live backend.

Usage:
    .venv/bin/python -m evals.run
    .venv/bin/python -m evals.run --base-url http://localhost:8000 --limit 5
    .venv/bin/python -m evals.run --out evals/report.json

Each case opens its own conversation (isolated history), sends the question
through the REST turn endpoint, and scores the reply with `evals.scoring`.
There is no A/B comparison: every run evaluates the current pipeline only.
Exit code is 1 when any case fails, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

from evals.scoring import CaseResult, score_case

DATASET_PATH = Path(__file__).with_name("dataset.jsonl")
REPORT_PATH = Path(__file__).with_name("report.json")


def load_dataset(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


async def run_case(client: httpx.AsyncClient, token: str, case: dict) -> CaseResult:
    headers = {"Authorization": f"Bearer {token}"}
    conversation = (
        await client.post("/chat/conversations", headers=headers)
    ).json()["id"]

    started = time.perf_counter()
    data: dict = {}
    error: str | None = None
    try:
        response = await client.post(
            f"/chat/conversations/{conversation}/messages",
            headers=headers,
            json={"conversation_id": conversation, "content": case["question"]},
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000.0

    bot_message = data.get("bot_message") or {}
    return score_case(
        case,
        answer=bot_message.get("content") or "",
        escalated=bool(data.get("escalate")),
        sources=bot_message.get("sources") or [],
        latency_ms=latency_ms,
        error=error,
    )


async def amain(args: argparse.Namespace) -> int:
    cases = load_dataset(Path(args.dataset))
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        cases = [case for case in cases if case.get("id") in wanted]
    if args.limit:
        cases = cases[: args.limit]

    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        auth = await client.post("/auth/anonymous", json={"first_name": "Eval Runner"})
        auth.raise_for_status()
        token = auth.json()["access_token"]

        results: list[CaseResult] = []
        for case in cases:
            result = await run_case(client, token, case)
            results.append(result)
            print(f"[{'PASS' if result.passed else 'FAIL'}] {result.case_id} "
                  f"({result.latency_ms:.0f} ms)")
            if not result.passed:
                for name in result.failed_checks:
                    print(f"        missed check: {name}")
                if result.error:
                    print(f"        error: {result.error}")

    passed = sum(1 for result in results if result.passed)
    latencies = [result.latency_ms for result in results if result.error is None]
    mean_ms = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n=== summary ===")
    print(f"passed: {passed}/{len(results)}")
    print(f"mean latency: {mean_ms:.0f} ms")

    report = {
        "base_url": args.base_url,
        "passed": passed,
        "total": len(results),
        "mean_latency_ms": round(mean_ms, 1),
        "cases": [result.to_dict() for result in results],
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out_path}")

    return 0 if passed == len(results) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hermes golden-set eval.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--out", default=str(REPORT_PATH))
    parser.add_argument("--only", default="", help="Comma-separated case ids to run.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases.")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(parse_args())))
