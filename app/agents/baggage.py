"""In-process baggage guidance for lost / delayed / unclaimed bags.

Detection is deterministic (regex + keyword cues), mirroring
`app.agents.flights`: the decision to show baggage guidance is auditable and
cheap, and the LLM only ever relays the rendered guidance rather than
inventing contact points. Unlike flight data this is not an external system,
so it stays in-process (see the plan's MCP scope).

Guidance covers the three places a passenger can find or trace a bag:

1. the arrival baggage claim hall (JKIA Terminal 1A),
2. the airport Lost & Found desk, and
3. online WorldTracer tracking.

The answer path then proactively offers a human agent if none of those
resolves the problem.
"""

from __future__ import annotations

import re

from app.config import settings

# A baggage-ish noun. "Where is the baggage claim?" has no separate problem
# cue but does contain "claim", so it still matches below.
_BAGGAGE_THEME = re.compile(
    r"\b(bag|bags|baggage|luggage|suitcase|carousel|reclaim|"
    r"world ?tracer|lost (?:and|&) found|lost property)\b",
    re.IGNORECASE,
)

# A word that signals a *problem* (as opposed to a general baggage question
# such as "what is my baggage allowance?").
_BAGGAGE_PROBLEM = re.compile(
    r"\b(lost|missing|delay(?:ed)?|didn'?t|did not|hasn'?t|hadn'?t|"
    r"not arriv(?:e|ed)|never (?:arrived|showed|turned up)|where|"
    r"trace|track|claim|reclaim|found|stolen|damaged|left behind|"
    r"world ?tracer|went missing|didn'?t (?:make|show))\b",
    re.IGNORECASE,
)


def detect_baggage_request(query: str) -> bool:
    """Whether `query` is about a lost, delayed, or unclaimed bag.

    A general baggage question ("what is my checked baggage allowance?") is
    left to the knowledge base; only a baggage noun *and* a problem cue (or an
    explicit "baggage claim" / "lost and found" phrase) triggers guidance.
    """
    text = query or ""
    return bool(_BAGGAGE_THEME.search(text) and _BAGGAGE_PROBLEM.search(text))


def guidance_lines() -> list[str]:
    """The three places to look, as short, reliable lines."""
    return [
        (
            f"Arrival baggage claim: {settings.baggage_arrival_claim}. Check "
            "here first - delayed bags often arrive on a later flight."
        ),
        (
            f"Airport Lost & Found desk: {settings.baggage_lost_and_found}. "
            "Staff there can file a report and give you a reference number."
        ),
        (
            f"Online tracking (WorldTracer): {settings.worldtracer_url} - "
            "you'll need the bag tag number or the reference on your baggage "
            "receipt."
        ),
    ]


def render_guidance() -> str:
    """Rendered guidance, injected into the QA system prompt as context."""
    return "\n".join(guidance_lines())
