"""Deterministic greeting / small-talk path.

The first thing a passenger sends is usually "hi" or "hello". Sending that to
the retrieval pipeline was a mistake: the knowledge base has nothing to say
about a greeting, and the QA prompt used to read "no context" as "cannot
help" — so the very first turn of a conversation came back as "I'm sorry, I
don't have that information ... would you like me to connect you with a human
agent?".

Greetings are answered here instead: no LLM call, no retrieval, no handoff.
Only a *standalone* greeting counts; "hi, I need to rebook my flight" has real
content and still goes through the normal pipeline.
"""

import re

_GREETING_RE = re.compile(
    r"^(?:hi|hey|hello|hallo|yo|yeah|sup|hiya|greetings|good\s+(?:morning|afternoon|"
    r"evening|day)|morning|afternoon|evening|habari|karibu|sawubona|howzit)\b"
    r"[\s!.,?]*$",
    re.IGNORECASE,
)

_THANKS_RE = re.compile(
    r"^(?:thanks|thank\s+you|many\s+thanks|thank\s+u|ty|asante|cheers|"
    r"great|perfect|awesome|nice\s+one|cool|okay|ok|okay\s+thanks|"
    r"sure|alright|got\s+it|understood|noted|will\s+do)\b[\s!.,?]*$",
    re.IGNORECASE,
)

GREETING_ANSWER = (
    "Hi! I'm hermes, the Kenya Airways support assistant. I can check your "
    "flight, answer baggage and check-in questions, and rebook a disrupted "
    "flight for you right here.\n\n"
    "What can I help you with today?"
)

THANKS_ANSWER = (
    "Happy to help! If anything else comes up with your trip, just ask here."
)

# Starter prompts offered on a greeting. The frontend keeps its own copy for
# the empty state; these are the ones that follow a bot greeting.
STARTER_OPTIONS = [
    {"label": "Check my flight", "value": "How can I check if my flight is delayed?"},
    {"label": "Book a flight", "value": "I want to book a flight."},
    {"label": "Rebook a flight", "value": "My flight was cancelled. How do I rebook?"},
    {"label": "Baggage allowance", "value": "What is my checked baggage allowance?"},
    {"label": "Lost baggage", "value": "My bag didn't arrive. What should I do?"},
]


def is_greeting(message: str) -> bool:
    """Whether `message` is a standalone greeting."""
    text = (message or "").strip()
    if not text or len(text) > 40:
        return False
    return bool(_GREETING_RE.match(text))


def is_thanks(message: str) -> bool:
    """Whether `message` is a standalone thank-you / acknowledgement."""
    text = (message or "").strip()
    if not text or len(text) > 40:
        return False
    return bool(_THANKS_RE.match(text))


def greeting_reply(message: str) -> tuple[str, list[dict]] | None:
    """The canned reply for a greeting, or None when the message is not one."""
    if is_greeting(message):
        return GREETING_ANSWER, [dict(option) for option in STARTER_OPTIONS]
    if is_thanks(message):
        return THANKS_ANSWER, []
    return None
