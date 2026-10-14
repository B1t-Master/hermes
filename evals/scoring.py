"""Deterministic scoring for the hermes eval harness.

No LLM judge and no A/B variants: each golden case declares which terms the
answer must contain, must avoid, and whether the turn should escalate or cite
a source. The harness scores those checks exactly, so a run is reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Models render numbers with non-breaking/narrow spaces ("7\u00a0days"); fold
# every whitespace variant to a plain space so term matching is stable.
_WHITESPACE = re.compile(r"[\s\u00a0\u202f]+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", (text or "").lower()).strip()


@dataclass
class CaseResult:
    case_id: str
    question: str
    passed: bool
    checks: list[tuple[str, bool]]
    answer: str
    escalated: bool
    cited: bool
    latency_ms: float
    error: str | None = None

    @property
    def failed_checks(self) -> list[str]:
        return [name for name, ok in self.checks if not ok]

    def to_dict(self) -> dict:
        return {
            "id": self.case_id,
            "question": self.question,
            "passed": self.passed,
            "escalated": self.escalated,
            "cited": self.cited,
            "latency_ms": round(self.latency_ms, 1),
            "failed_checks": self.failed_checks,
            "error": self.error,
            "answer": self.answer,
        }


def score_case(
    case: dict,
    *,
    answer: str,
    escalated: bool,
    sources: list | None,
    latency_ms: float = 0.0,
    error: str | None = None,
) -> CaseResult:
    """Score one golden case against the observed turn."""
    text = _normalize(answer)
    checks: list[tuple[str, bool]] = []

    for term in case.get("expect_contains") or []:
        checks.append((f"contains:{term}", _normalize(term) in text))

    any_terms = case.get("expect_any") or []
    if any_terms:
        checks.append(
            (f"any:{'/'.join(any_terms)}", any(_normalize(term) in text for term in any_terms))
        )

    for term in case.get("expect_none") or []:
        checks.append((f"absent:{term}", _normalize(term) not in text))

    expected_escalate = case.get("expect_escalate")
    if expected_escalate is not None:
        checks.append(("escalate", bool(escalated) == bool(expected_escalate)))

    expected_cite = case.get("expect_cite")
    if expected_cite is not None:
        checks.append(("cite", bool(sources) == bool(expected_cite)))

    passed = error is None and all(ok for _, ok in checks)
    return CaseResult(
        case_id=case.get("id", "?"),
        question=case.get("question", ""),
        passed=passed,
        checks=checks,
        answer=answer,
        escalated=bool(escalated),
        cited=bool(sources),
        latency_ms=latency_ms,
        error=error,
    )
