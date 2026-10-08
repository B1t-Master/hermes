"""Intent detection: deterministic keyword layer + LLM enum classifier.

Layer 1 (keyword): hard safety triggers match instantly and always escalate.
Safety-critical routing must be auditable and cannot depend on a model call.

Layer 2 (LLM): one small structured call for the nuanced intents. Fails open
to `routine` on any error; the keyword layer still protects the safety path.
"""

import logging
import re
from dataclasses import dataclass

from app.agents.llm import get_llm

logger = logging.getLogger(__name__)

# The only enum the classifier may return.
INTENTS = ("routine", "compensation", "safety", "complaint", "human_request")

# Layer 1: substring matches on the lowercased message. Kept deliberately
# narrow: genuine emergencies/danger only. Nuanced cases (e.g. delayed
# baggage complaints) belong to the LLM layer.
HARD_SAFETY_TERMS = (
    "emergency",
    "bomb",
    "terrorist",
    "hijack",
    "hijacking",
    "gun",
    "weapon",
    "shooting",
    "fire on board",
    "medical emergency",
    "unconscious",
    "heart attack",
    "chest pain",
    "dying",
    "death threat",
    "engine failure",
    "emergency landing",
)

_CLASSIFY_PROMPT = """\
You are an intent classifier for Kenya Airways passenger support messages.
Pick exactly ONE intent for the user message:

- routine: factual info requests (baggage rules, check-in, fares, schedules, status)
- compensation: requests/demands for money, refunds, reimbursement, or vouchers
- safety: security threats, medical situations, danger, emergencies
- complaint: negative feedback about service, staff, or experience
- human_request: asks to speak to a person

Reply with ONLY the intent word, nothing else.

User message: {query}
Intent:"""


@dataclass
class IntentResult:
    intent: str  # one of INTENTS
    source: str  # "keyword" | "llm" | "default"
    matched_term: str | None = None  # keyword layer only


def keyword_hard_trigger(query: str) -> str | None:
    """The safety term found in `query`, if any (case-insensitive substring)."""
    haystack = (query or "").lower()
    for term in HARD_SAFETY_TERMS:
        if term in haystack:
            return term
    return None


def parse_llm_intent(raw: str) -> str | None:
    """Extract a valid intent word from raw LLM output; None when unparseable."""
    if not raw:
        return None
    words = re.findall(r"[a-z_]+", raw.lower())
    for word in words:
        if word in INTENTS:
            return word
    return None


async def classify_intent(query: str) -> IntentResult:
    matched = keyword_hard_trigger(query)
    if matched:
        return IntentResult(intent="safety", source="keyword", matched_term=matched)

    llm = get_llm()
    if not llm.is_configured:
        return IntentResult(intent="routine", source="default")

    try:
        raw = await llm.complete(
            [{"role": "user", "content": _CLASSIFY_PROMPT.format(query=(query or "").strip()[:2000])}],
            temperature=0.0,
            max_tokens=512,
        )
    except Exception:
        logger.exception("Intent classification failed; failing open to routine")
        return IntentResult(intent="routine", source="default")

    intent = parse_llm_intent(raw)
    if intent is None:
        logger.warning("Unparseable intent output %r; failing open to routine", raw)
        return IntentResult(intent="routine", source="default")
    return IntentResult(intent=intent, source="llm")
