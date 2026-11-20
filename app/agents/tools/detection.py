"""Deterministic detection of booking and rebooking intent.

These keyword layers decide whether the LLM should be offered the booking or
rebooking tools on a turn. They are deliberately independent of any model call.
"""

from __future__ import annotations

import re

from services import booking

# Action phrases that mark a genuine "move me to another flight" request.
# Purely informational questions ("is self-rebooking free?", "can I rebook a
# disrupted flight?") must NOT match: the model should answer those from the
# knowledge base, not act. The model still chooses the tool and its arguments.
REBOOK_PATTERNS = (
    r"\bre-?book me\b",
    r"\bre-?book (?:my|our|the) (?:flight|booking|ticket|itinerary|reservation)\b",
    r"\b(?:please|can you|could you|would you) re-?book\b",
    r"(?<!do )\bi (?:want|need|would like|'d like) to re-?book\b",
    r"\bchange (?:my|our|the) (?:flight|booking|ticket|itinerary|reservation)\b",
    r"\bmove (?:me|us|my flight) to\b",
    r"\breschedule (?:my|our|the) flight\b",
    r"\b(?:another|alternative|different) flight\b",
    r"\bnext available flight\b",
)
_REBOOK_RE = re.compile("|".join(REBOOK_PATTERNS), re.IGNORECASE)

# A follow-up turn that supplies the booking reference the rebooking needs.
# Accepts digit-only references ("1234567890"), hyphenated alphanumeric ones
# ("HRM-9B4770", "KQ-123456", "TEST-0001") and letters+digits ("KQ123456").
# A hyphen or a digit run is required so ordinary words ("baggage") do not
# masquerade as booking references.
_TICKET_HINT = re.compile(r"\b(ticket|booking|reference|pnr|e-?ticket)\b", re.IGNORECASE)
_TICKET_NUMBER = re.compile(
    r"\b(?:\d{6,14}|[A-Z]{2,4}-[A-Z0-9]{4,}|[A-Z]{2,4}\d{4,})\b", re.IGNORECASE
)

# A follow-up turn that supplies the new travel date ("tomorrow", "tomorrow
# afternoon", "day after tomorrow", "2025-04-20", "next Friday", ...).
_RELATIVE_DATE = re.compile(
    r"\b(today|tonight|tomorrow|day after tomorrow|overmorrow|next day|"
    r"next week|next month|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Flight numbers like KQ100, KQ 720 - used to detect a passenger choosing one
# of the rebooking alternatives presented by the assistant.
_FLIGHT_NO = re.compile(r"\bKQ\s?\d{2,4}\b", re.IGNORECASE)

# What our own assistant says while walking the passenger through a rebooking,
# used to keep the tools available on the follow-up turns (the passenger's
# answer alone often looks like plain text: "HRM-9B4770", "tomorrow").
_ASSISTANT_REBOOK_PROMPT = re.compile(
    r"booking reference|e-?ticket number|new travel date|"
    r"what.{0,20}date|which date|re-?book",
    re.IGNORECASE,
)

# A genuine new-booking request. Word-boundary "book" deliberately does not
# match "rebook"; rebooking is detected first and takes precedence.
BOOKING_PATTERNS = (
    r"\bbook (?:me |us )?(?:a |an )?(?:flight|ticket|seat|trip)\b",
    r"\bbook (?:a |an )?(?:flight|ticket|seat|trip) (?:to|from)\b",
    r"\bi (?:want|need|would like|'d like) to (?:book|fly|travel)\b",
    r"\b(?:find|search|show)(?: me)? (?:some |any |available )?flights?\b",
    r"\bflights? (?:from|to)\b",
    r"\b(?:one[- ]way|return|round[- ]trip)\b[^.]{0,40}\b(?:flight|ticket)\b",
    r"\b(?:flight|ticket) (?:to|from)\b",
    r"\bavailability (?:for|on)\b",
)
_BOOKING_RE = re.compile("|".join(BOOKING_PATTERNS), re.IGNORECASE)

# What our own assistant says while walking the passenger through a booking,
# used to keep the tools available on follow-up turns that are just a city or
# a date ("Mombasa", "next Friday").
_ASSISTANT_BOOKING_PROMPT = re.compile(
    r"payment card|which date|what date|travel date|departure date|"
    r"where (?:are|would) you (?:flying|like to fly)|flying from|flying to|"
    r"which route|search(?:ing)? flights|book (?:you|a flight)|"
    r"fare|m-?pesa|phone number|mobile number",
    re.IGNORECASE,
)

# A question about the booking the passenger already has ("what's my ticket
# number?") is not a request to rebook, even mid-flow: answering it from
# memory is the whole point. Re-entering the tool flow made the bot ask for
# the booking reference again after it had already rebooked with it.
_OWN_BOOKING_QUESTION = re.compile(
    r"\b(?:what(?:'s| is| was)?|whats|which|tell me|remind me|do you have|"
    r"do you know|can you tell me|give me)\b[^?]{0,40}?\b(?:my|our)\b"
    r"[^?]{0,30}?\b(?:ticket(?: number)?|booking reference|reference|pnr|"
    r"e-?ticket(?: number)?|confirmation(?: reference)?|flight number)\b",
    re.IGNORECASE,
)


def _has_rebook_intent(text: str) -> bool:
    return bool(_REBOOK_RE.search(text))


def _has_booking_intent(text: str) -> bool:
    return bool(_BOOKING_RE.search(text))


def _in_booking_flow(history: list[dict] | None) -> bool:
    recent = (history or [])[-8:]
    if any(
        m.get("role") == "user" and _has_booking_intent(m.get("content", ""))
        for m in recent
    ):
        return True
    last_assistant = next(
        (m for m in reversed(recent) if m.get("role") == "assistant"), None
    )
    return bool(
        last_assistant
        and _ASSISTANT_BOOKING_PROMPT.search(last_assistant.get("content", ""))
    )


def detect_booking_request(query: str, history: list[dict] | None = None) -> bool:
    """Whether to offer the flight-search/booking tools on this turn.

    Rebooking wins over booking; a question about the passenger's own booking
    is answered from memory; follow-up turns stay in the flow while the
    assistant is collecting a route or date.
    """
    if _OWN_BOOKING_QUESTION.search(query):
        return False
    if _has_rebook_intent(query):
        return False
    if _has_booking_intent(query):
        return True
    if not history or not _in_booking_flow(history):
        return False
    if booking.parse_route(query) or _looks_like_date(query):
        return True
    # Short answers during a booking flow (city, date, phone number) keep the
    # tools available so the passenger can complete the booking.
    return len(query.split()) <= 8


def _looks_like_ticket(text: str) -> bool:
    return bool(_TICKET_HINT.search(text) or _TICKET_NUMBER.search(text))


def _looks_like_date(text: str) -> bool:
    return bool(_RELATIVE_DATE.search(text) or _ISO_DATE.search(text))


def _in_rebook_flow(history: list[dict] | None) -> bool:
    """Whether a rebooking conversation is already underway.

    Keeps the tools available from the moment the passenger asks to rebook
    until the flow completes, so follow-up turns that contain only a booking
    reference or a date ("HRM-9B4770", "tomorrow afternoon") stay on the
    rebook path instead of falling back to the knowledge-base answer.
    """
    recent = (history or [])[-6:]
    if any(
        m.get("role") == "user" and _has_rebook_intent(m.get("content", ""))
        for m in recent
    ):
        return True
    last_assistant = next(
        (m for m in reversed(recent) if m.get("role") == "assistant"), None
    )
    return bool(
        last_assistant
        and _ASSISTANT_REBOOK_PROMPT.search(last_assistant.get("content", ""))
    )


def detect_rebook_request(query: str, history: list[dict] | None = None) -> bool:
    """Whether to offer the rebook tools on this turn.

    True for a direct rebook ask, or while a rebooking conversation is already
    underway and this turn supplies the booking reference or the new date (so
    the multi-turn flow keeps the tools available). Unrelated follow-ups stay
    on the plain path, and a question about the passenger's own booking
    details ("what's my ticket number?") is answered from memory instead.
    """
    if _OWN_BOOKING_QUESTION.search(query):
        return False
    if _has_rebook_intent(query):
        return True
    if not history:
        return False
    if not (
        _looks_like_ticket(query) or _looks_like_date(query) or _FLIGHT_NO.search(query)
    ):
        return False
    return _in_rebook_flow(history)
