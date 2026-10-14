"""Run the hermes golden-set evaluation against a live backend.

Usage:
    .venv/bin/python -m evals.run                 # full suite, 4 cases at once
    .venv/bin/python -m evals.run --fast          # quick subset for a smoke check
    .venv/bin/python -m evals.run --concurrency 8
    .venv/bin/python -m evals.run --only rebook-full-flow
    .venv/bin/python -m evals.run --out evals/report.json

Each case opens its own conversation (isolated history) and may send follow-up
turns (``follow_ups``) before scoring. Cases run concurrently up to
``--concurrency``; the backend is async, so this cuts wall-clock time several
fold versus a serial run. There is no A/B comparison: every run evaluates the
current pipeline only. Exit code is 1 when any case fails, so CI can gate on it.
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
    """Send the case's turns in one conversation and score the bot replies."""
    headers = {"Authorization": f"Bearer {token}"}
    conversation = (await client.post("/chat/conversations", headers=headers)).json()["id"]
    prompts = [case["question"], *(case.get("follow_ups") or [])]

    started = time.perf_counter()
    answers: list[str] = []
    sources: list = []
    escalated = False
    error: str | None = None
    try:
        for content in prompts:
            response = await client.post(
                f"/chat/conversations/{conversation}/messages",
                headers=headers,
                json={"conversation_id": conversation, "content": content},
            )
            response.raise_for_status()
            data = response.json()
            bot_message = data.get("bot_message") or {}
            if bot_message.get("content"):
                answers.append(bot_message["content"])
            if bot_message.get("sources"):
                sources = bot_message["sources"]
            escalated = escalated or bool(data.get("escalate"))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000.0

    return score_case(
        case,
        answer="\n".join(answers),
        escalated=escalated,
        sources=sources,
        latency_ms=latency_ms,
        error=error,
    )


def select_cases(args: argparse.Namespace) -> list[dict]:
    cases = load_dataset(Path(args.dataset))
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        cases = [case for case in cases if case.get("id") in wanted]
    if args.fast:
        cases = [case for case in cases if case.get("fast")]
    if args.limit:
        cases = cases[: args.limit]
    return cases


async def amain(args: argparse.Namespace) -> int:
    cases = select_cases(args)
    if not cases:
        print("no cases selected")
        return 1

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        auth = await client.post("/auth/anonymous", json={"first_name": "Eval Runner"})
        auth.raise_for_status()
        token = auth.json()["access_token"]

        started = time.perf_counter()

        async def run_one(index: int, case: dict) -> tuple[int, CaseResult]:
            async with semaphore:
                return index, await run_case(client, token, case)

        tasks = [asyncio.create_task(run_one(i, case)) for i, case in enumerate(cases)]
        results: list[CaseResult | None] = [None] * len(cases)
        for finished in asyncio.as_completed(tasks):
            index, result = await finished
            results[index] = result
            print(f"[{'PASS' if result.passed else 'FAIL'}] {result.case_id} "
                  f"({result.latency_ms:.0f} ms)")
            if not result.passed:
                for name in result.failed_checks:
                    print(f"        missed check: {name}")
                if result.error:
                    print(f"        error: {result.error}")

    ordered = [result for result in results if result is not None]
    wall_ms = (time.perf_counter() - started) * 1000.0
    passed = sum(1 for result in ordered if result.passed)
    latencies = [result.latency_ms for result in ordered if result.error is None]
    mean_ms = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n=== summary ===")
    print(f"passed: {passed}/{len(ordered)}")
    print(f"mean case latency: {mean_ms:.0f} ms")
    print(f"wall clock: {wall_ms:.0f} ms (concurrency {max(1, args.concurrency)})")

    report = {
        "base_url": args.base_url,
        "passed": passed,
        "total": len(ordered),
        "mean_latency_ms": round(mean_ms, 1),
        "wall_clock_ms": round(wall_ms, 1),
        "concurrency": max(1, args.concurrency),
        "cases": [result.to_dict() for result in ordered],
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out_path}")

    return 0 if passed == len(ordered) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hermes golden-set eval.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--out", default=str(REPORT_PATH))
    parser.add_argument("--only", default="", help="Comma-separated case ids to run.")
    parser.add_argument(
        "--fast", action="store_true", help="Run only cases tagged fast (quick smoke)."
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Max cases in flight at once."
    )
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases.")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(parse_args())))
