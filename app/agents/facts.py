"""Structured conversation memory: extract durable slots from passenger turns.

Slots persist on ``Conversation.facts`` so the bot never re-asks for something
the passenger already told it — a booking reference, an email, a phone number,
or a travel date. Extraction is deterministic (regex), cheap, and auditable:
the LLM is not trusted to remember, and a known fact survives even after the
raw message falls out of the recent-history window.

Two rules keep the memory honest:

1. Only the passenger's own words can set ``ticket_number``. The bot's
   rebooking confirmation quotes a *confirmation* reference ("Confirmation
   reference HRM-D344AC"); letting that land in ``ticket_number`` silently
   replaced the passenger's real e-ticket number, so "what is my ticket
   number?" answered with the wrong code. Assistant messages may only fill
   slots the passenger has not provided, and never overwrite them.
2. A successful ``rebook_flight`` result is authoritative: ``apply_rebook_result``
   merges the tool's ticket / reference / flight / date straight into memory
   instead of hoping the model repeated them in prose.
"""

from __future__ import annotations

import re

# --- text normalisation ---------------------------------------------------

# Models and phones emit non-breaking hyphens (HRM-D344AC), narrow no-break
# spaces and non-breaking spaces. Every pattern below assumes plain ASCII
# punctuation, so the text is folded before matching.
_DASHES = re.compile(r"[\u2010-\u2015\u2212]")
_SPACES = re.compile(r"[\u00a0\u202f\u2007]")


def normalise(text: str) -> str:
    """Fold unicode dashes/spaces to ASCII so the patterns match reliably."""
    if not text:
        return ""
    folded = _DASHES.sub("-", text)
    folded = _SPACES.sub(" ", folded)
    return folded.replace("\r\n", "\n").replace("\r", "\n")


# 13-digit e-ticket numbers and long numeric booking references.
_LONG_DIGITS = re.compile(r"\b\d{10,14}\b")
# Hyphenated references such as "HRM-9B4770" or "KQ-123456".
_HYPHEN_REF = re.compile(r"\b[A-Z]{2,4}-[A-Z0-9]{4,}\b", re.IGNORECASE)
# Compact references without a hyphen ("HRM9B4770", "KQ123456").
_COMPACT_REF = re.compile(r"\b[A-Z]{2,4}\d{4,12}\b", re.IGNORECASE)
# A reference supplied right after a hint word: "reference is ABC123".
# The lookahead requires a digit, so the hint word itself ("reference") is not
# mistaken for the reference.
_REF_HINT = re.compile(
    r"\b(?:booking|reference|pnr|e-?ticket|ticket(?:\s+number)?)\b\s*(?:is|:)?\s*"
    r"(?=[A-Z0-9-]*\d)([A-Z0-9][A-Z0-9-]{4,14})",
    re.IGNORECASE,
)
# A *confirmation* reference: the code the rebooking hands back. It is not the
# passenger's booking reference, so it is kept in its own slot.
_CONFIRMATION_REF = re.compile(
    r"\b(?:confirmation|rebooking)\s+(?:reference|number|code)\b\s*(?:is|:)?\s*"
    r"([A-Z0-9][A-Z0-9-]{4,19})"
    r"|\breference\s+number\b\s*(?:is|:)?\s*([A-Z0-9][A-Z0-9-]{4,19})",
    re.IGNORECASE,
)
# "your new reference is HRM-D344AC" — loose, but only inside a sentence that
# mentions a confirmation, so a plain booking reference is not mistaken.
_CONFIRMATION_LOOSE = re.compile(
    r"\b(?:confirmation|rebooking|re-booking)\b[^.\n]{0,32}?"
    r"\b([A-Z]{2,4}-[A-Z0-9]{4,})\b",
    re.IGNORECASE,
)
# A WorldTracer / bag-tag reference is a tracking reference, not a booking one.
_TRACKING_HINT = re.compile(
    r"\b(?:world ?tracer|tracking (?:reference|number|ref)|bag ?tag)\b",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Kenyan mobile numbers (07xx / 01xx, or +254...) plus generic internationals.
_PHONE = re.compile(
    r"(?:\+?254|0)\s?[17]\d{2}[\s-]?\d{3}[\s-]?\d{3}\b|\+\d[\d\s-]{8,15}\d"
)
_RELATIVE_DATE = re.compile(
    r"\b(today|tonight|tomorrow|day after tomorrow|overmorrow|next day|next week|"
    r"next month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Flight numbers like KQ100, KQ 720
_FLIGHT_NO = re.compile(r"\bKQ\s?\d{2,4}\b", re.IGNORECASE)

# The memory slots we persist.
FACT_KEYS = (
    "ticket_number",
    "confirmation_ref",
    "email",
    "phone",
    "new_date",
    "tracking_ref",
    "flight_no",
    "new_flight_no",
    "route",
    "human_declined",
    "booking_pnr",
    "booking_flight",
    "booking_route",
    "booking_date",
    "booking_amount",
    "booking_status",
    "payment_id",
    "eticket_number",
)

# The passenger declining a human handoff ("no thanks, I'll continue").
_HUMAN_DECLINE = re.compile(
    r"\b(?:no\s*,?\s*thanks|no\s+thank\s+you|not\s+now|i(?:'ll|\s+will)\s+continue|"
    r"keep\s+going|i(?:'m|\s+am)\s+fine|that(?:'s|\s+is)\s+fine|"
    r"don(?:'t|\s+not)\s+need\s+(?:a|an)\s+human|not\s+to\s+connect)\b",
    re.IGNORECASE,
)


def _normalise_phone(raw: str) -> str:
    return re.sub(r"[\s-]", "", raw)


def detect_human_decline(message: str) -> bool:
    """Whether the passenger declined a human handoff on this turn."""
    return bool(_HUMAN_DECLINE.search(normalise(message or "")))


def _confirmation_tokens(text: str) -> set[str]:
    """Confirmation references mentioned in `text` (uppercased)."""
    tokens: set[str] = set()
    for match in _CONFIRMATION_REF.finditer(text):
        token = match.group(1) or match.group(2)
        if token:
            tokens.add(token.upper())
    for match in _CONFIRMATION_LOOSE.finditer(text):
        tokens.add(match.group(1).upper())
    return tokens


def _looks_like_phone(candidate: str) -> bool:
    """Guard: a Kenyan mobile number is never a booking reference."""
    compact = candidate.replace(" ", "").replace("-", "")
    return bool(
        re.fullmatch(r"0\d{9}", compact) or re.fullmatch(r"\+254\d{9}", compact)
    )


def _ticket_token(text: str, confirmations: set[str]) -> str | None:
    """The booking reference in `text`, if there is one."""
    hinted = _REF_HINT.search(text)
    if hinted:
        token = hinted.group(1)
    else:
        token = None
        for pattern in (_LONG_DIGITS, _HYPHEN_REF, _COMPACT_REF):
            match = pattern.search(text)
            if not match:
                continue
            candidate = match.group(0)
            if _looks_like_phone(candidate):
                continue
            token = candidate
            break
    if not token:
        return None
    token = token.upper()
    # A confirmation reference is not the passenger's booking reference.
    if token in confirmations:
        return None
    return token


def extract_facts(
    message: str,
    existing: dict | None = None,
    *,
    source: str = "passenger",
) -> dict:
    """Merge newly-seen slots in `message` into a copy of `existing`.

    A slot already known is left untouched when this turn does not mention it,
    so memory accumulates across turns instead of being overwritten.

    `source` is ``"passenger"`` for the passenger's own words and
    ``"assistant"`` for the bot's (or a human agent's) messages. Assistant
    messages may fill empty slots but never overwrite the passenger's, and a
    confirmation reference in them is never mistaken for a ticket number.
    """
    facts = dict(existing or {})
    text = normalise(message or "")
    from_passenger = source == "passenger"

    email = _EMAIL.search(text)
    if email and (from_passenger or not facts.get("email")):
        facts["email"] = email.group(0).lower()

    phone = _PHONE.search(text)
    if phone and (from_passenger or not facts.get("phone")):
        facts["phone"] = _normalise_phone(phone.group(0))

    # Confirmation references from rebooking results ("Confirmation reference
    # HRM-D344AC"). Kept apart from the booking reference on purpose.
    confirmations = _confirmation_tokens(text)
    for token in confirmations:
        facts["confirmation_ref"] = token
        break

    token = _ticket_token(text, confirmations)
    if token:
        if _TRACKING_HINT.search(text):
            # A bag-tag / WorldTracer number is a tracking reference.
            if from_passenger or not facts.get("tracking_ref"):
                facts["tracking_ref"] = token
        elif from_passenger:
            facts["ticket_number"] = token
        elif not facts.get("ticket_number"):
            # The bot repeated the booking reference; trust it only when the
            # passenger has not given one themselves.
            facts["ticket_number"] = token

    date_match = _ISO_DATE.search(text) or _RELATIVE_DATE.search(text)
    if date_match and (from_passenger or not facts.get("new_date")):
        facts["new_date"] = date_match.group(0).lower()

    # Capture flight numbers
    flights = _FLIGHT_NO.findall(text)
    if flights:
        # If context says "new flight" or "rebooked to", capture as new_flight_no
        if re.search(r"\b(?:rebooked to|re-booked to|new flight|now on|moved to)\b", text, re.IGNORECASE):
            if from_passenger or not facts.get("new_flight_no"):
                facts["new_flight_no"] = flights[0].upper().replace(" ", "")
        elif "new_flight_no" not in facts and "flight_no" not in facts:
            facts["flight_no"] = flights[0].upper().replace(" ", "")
        elif "new_flight_no" not in facts:
            facts["new_flight_no"] = flights[0].upper().replace(" ", "")

    return facts


def apply_rebook_result(facts: dict | None, result: dict) -> dict:
    """Merge an authoritative `rebook_flight` result into the memory slots.

    The tool knows the ticket, the new confirmation reference, the flight and
    the travel date exactly. Writing them straight into memory means the next
    question ("what is my ticket number?") is answered from the booking, not
    from whatever the model happened to repeat in its last sentence.
    """
    merged = dict(facts or {})
    if not result or not result.get("rebooked"):
        return merged

    ticket = result.get("ticket_number")
    if ticket:
        merged["ticket_number"] = str(ticket).upper()
    reference = result.get("reference")
    if reference:
        merged["confirmation_ref"] = str(reference).upper()
    flight_no = result.get("flight_no")
    if flight_no:
        merged["new_flight_no"] = str(flight_no).upper().replace(" ", "")
    new_date = result.get("date")
    if new_date:
        merged["new_date"] = str(new_date)
    route = result.get("route")
    if route:
        merged["route"] = str(route)
    return merged


def apply_booking_result(facts: dict | None, result: dict) -> dict:
    """Merge an authoritative booking/payment result into memory.

    ``create_booking`` and ``issue_ticket`` know the PNR, flight, amount and
    e-ticket exactly. Writing them into the slots means later questions about
    the new booking are answered from the record, not from the model's prose.
    """
    merged = dict(facts or {})
    if not result:
        return merged

    pnr = result.get("pnr")
    if pnr:
        merged["booking_pnr"] = str(pnr).upper()
        merged["ticket_number"] = str(pnr).upper()
    flight_no = result.get("flight_no")
    if flight_no:
        merged["booking_flight"] = str(flight_no).upper().replace(" ", "")
    route = result.get("route")
    if not route and result.get("origin") and result.get("destination"):
        route = f"{result['origin']}-{result['destination']}"
    if route:
        merged["booking_route"] = str(route)
    travel_date = result.get("date")
    if travel_date:
        merged["booking_date"] = str(travel_date)
    amount = result.get("amount")
    if amount is not None:
        merged["booking_amount"] = amount
    status = result.get("status") or result.get("payment_status")
    if status:
        merged["booking_status"] = str(status)
    payment_id = result.get("payment_id")
    if payment_id:
        merged["payment_id"] = str(payment_id)
    eticket = result.get("eticket_number")
    if eticket:
        merged["eticket_number"] = str(eticket)
    return merged
