"""LLM function calling for the hermes agent.

Tool schemas are OpenAI-compatible. Flight tools go through the
`hermes-flights` MCP server (`app.agents.mcp_client`); rebooking routes through
the `hermes-ops` MCP server (with a direct fallback to `services/rebook.py`).
Booking/rebooking/tracking are the MCP-scoped operations per the plan.
Retrieval, profile lookups, baggage guidance, and escalation stay in-process.

`run_tool_turn` drives the model/tool loop: the model may call tools, results
are fed back as `tool` messages, and the loop returns once the model replies
with content instead of another tool call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from sqlalchemy import select

from app.agents.llm import LLM, get_llm
from app.agents.mcp_client import get_flights_client, get_ops_client
from app.config import settings
from app.db import SessionLocal
from app.models import Booking
from services import booking, email_real, rebook

logger = logging.getLogger(__name__)

# Cap on tool-calling rounds before we ask once more for a plain answer.
# Booking needs search -> create -> answer, hence three rounds.
MAX_TOOL_ROUNDS = 3

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


def _has_rebook_intent(text: str) -> bool:
    return bool(_REBOOK_RE.search(text))


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


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_flight_status",
            "description": (
                "Look up the current status of one Kenya Airways flight "
                "(on time, delayed, cancelled) and its scheduled times."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {
                        "type": "string",
                        "description": 'Flight number such as "KQ100".',
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date YYYY-MM-DD; defaults to today.",
                    },
                },
                "required": ["flight_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_delays",
            "description": (
                "List Kenya Airways flights delayed by 15 minutes or more, "
                "optionally filtered to one route."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "string",
                        "description": 'Optional route "NBO-LHR" or a single airport code.',
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date YYYY-MM-DD; defaults to today.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rebook_flight",
            "description": (
                "Rebook a passenger onto an available flight. Requires the "
                "passenger's booking reference or e-ticket number and the new "
                "travel date. When no flight number is supplied, returns the "
                "available alternatives for the passenger to choose from. "
                "When a flight number is supplied, confirms the rebooking. "
                "Emails a confirmation only when the booking has an email on "
                "file; ask for one if missing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {
                        "type": "string",
                        "description": (
                            "Flight number chosen from the alternatives, e.g. "
                            '"KQ444". Omit to see the available alternatives first.'
                        ),
                    },
                    "ticket_number": {
                        "type": "string",
                        "description": (
                            "The passenger's booking reference or e-ticket number. "
                            "If the passenger has not provided it yet, ask for it "
                            "first instead of calling this tool."
                        ),
                    },
                    "new_date": {
                        "type": "string",
                        "description": (
                            "The passenger's new travel date, e.g. 'tomorrow', "
                            "'tomorrow afternoon', 'day after tomorrow' or "
                            "YYYY-MM-DD."
                        ),
                    },
                    "passenger_name": {
                        "type": "string",
                        "description": "Passenger name, when known.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the flight is being rebooked (delay, cancellation, schedule change).",
                    },
                },
                "required": ["ticket_number", "new_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_rebook",
            "description": (
                "Confirm a rebooking: update the existing booking with the new "
                "flight and date, then email the new e-ticket. No payment is "
                "needed because demo fares are constant for rebooks. Call this "
                "after rebook_flight returns the alternatives and the passenger "
                "has chosen one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_number": {
                        "type": "string",
                        "description": "The passenger's booking reference or e-ticket number.",
                    },
                    "new_date": {
                        "type": "string",
                        "description": "The new travel date, e.g. 'tomorrow' or YYYY-MM-DD.",
                    },
                    "flight_no": {
                        "type": "string",
                        "description": 'Flight number chosen from the alternatives, e.g. "KQ444".',
                    },
                    "email": {
                        "type": "string",
                        "description": "Contact email for the new e-ticket (optional).",
                    },
                    "passenger_name": {
                        "type": "string",
                        "description": "Passenger name, when known.",
                    },
                },
                "required": ["ticket_number", "new_date", "flight_no"],
            },
        },
    },
]


BOOKING_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": (
                "Search Kenya Airways flights for one route and travel date. "
                "Use this before create_booking; never invent flights or fares. "
                "Origin and destination may be city names or IATA codes, and "
                "the date may be natural language ('tomorrow', '12 December') "
                "or YYYY-MM-DD."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": 'Departure city or code, e.g. "Nairobi" or "NBO".',
                    },
                    "destination": {
                        "type": "string",
                        "description": 'Arrival city or code, e.g. "Mombasa" or "MBA".',
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date, e.g. 'tomorrow', '12 December' or '2026-12-12'.",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_booking",
            "description": (
                "Create a held booking for a flight returned by search_flights. "
                "The reply renders an M-Pesa payment card that collects the "
                "passenger's number and starts the STK push, so never ask for "
                "the number in chat. Call this once you have a flight number "
                "and travel date. The result carries the PNR, fare, pending "
                "payment id, and an awaiting_phone flag."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {
                        "type": "string",
                        "description": 'Flight number from search_flights, e.g. "KQ444".',
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date for the chosen flight.",
                    },
                    "passenger_name": {
                        "type": "string",
                        "description": "Passenger name, when known.",
                    },
                    "passenger_email": {
                        "type": "string",
                        "description": "Passenger email for e-ticket delivery (optional).",
                    },
                    "passenger_phone": {
                        "type": "string",
                        "description": "Passenger phone number for M-Pesa payment (e.g. 0791351504).",
                    },
                },
                "required": ["flight_no", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_booking_email",
            "description": (
                "Add a passenger email to a paid booking and issue/email the "
                "e-ticket. Manual/administrative use only: the passenger booking "
                "flow ends at the payment card, and the agent dashboard issues "
                "the e-ticket after payment succeeds. Do not call this during a "
                "passenger conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pnr": {
                        "type": "string",
                        "description": "Booking reference (PNR) from create_booking.",
                    },
                    "email": {
                        "type": "string",
                        "description": "Passenger email address.",
                    },
                    "passenger_name": {
                        "type": "string",
                        "description": "Passenger name, when known.",
                    },
                },
                "required": ["pnr", "email"],
            },
        },
    },
]


def _canonical_name(name: str) -> str:
    """Drop any namespace prefix gpt-oss invents (e.g. ``repo_browser.rebook_flight``)."""
    return name.rsplit(".", 1)[-1].strip()


def _rebook(arguments: dict) -> dict:
    return rebook.rebook_flight(
        arguments.get("flight_no") or None,
        ticket_number=arguments.get("ticket_number"),
        new_date=arguments.get("new_date"),
        passenger_name=arguments.get("passenger_name"),
        reason=arguments.get("reason"),
    )


async def _lookup_ticket_email(ticket_number: str | None) -> str | None:
    """Destination email for a ticket, read from the booking's stored email.

    Previously used ``RebookEmailMap``; now reads the single source of truth
    ``Booking.passenger_email`` which is the email captured at booking time and
    also used for e-ticket delivery.
    """
    ticket = (ticket_number or "").strip()
    if not ticket:
        return None
    async with SessionLocal() as db:
        booking = (
            await db.execute(select(Booking).where(Booking.pnr == ticket.upper()))
        ).scalar_one_or_none()
    if booking is None:
        return None
    return booking.passenger_email


async def _send_rebook_confirmation(result: dict) -> None:
    """Email the confirmation when the ticket has an address on file.

    Reads the passenger email from the booking (the same email used for the
    e-ticket). The rebooking already succeeded, so delivery problems do not
    block the flow.
    """
    to = await _lookup_ticket_email(result.get("ticket_number"))
    if not to:
        result["email_sent"] = False
        result["email_note"] = (
            "No email address is on file for this booking reference, so no "
            "confirmation was sent."
        )
        return
    subject, body = rebook.confirmation_email(result)
    try:
        record = await asyncio.to_thread(
            email_real.send_email,
            to,
            subject,
            body,
            sender=settings.resend_from_email,
            api_key=settings.resend_api_key,
            outbox_dir=settings.email_outbox_dir,
        )
    except Exception:
        logger.exception("Rebook confirmation email failed for %s", result.get("reference"))
        return
    result["email_sent"] = True
    result["email_to"] = to
    result["email_provider"] = record.get("provider")
    result["email_sent_at"] = record.get("sent_at")


def _run_tool(name: str, arguments: dict) -> dict:
    name = _canonical_name(name)
    if name == "rebook_flight":
        return _rebook(arguments)
    if name == "confirm_rebook":
        return asyncio.run(
            booking.confirm_rebook(
                arguments.get("ticket_number", ""),
                arguments.get("new_date", ""),
                arguments.get("flight_no", ""),
                email=arguments.get("email") or None,
                passenger_name=arguments.get("passenger_name") or None,
            )
        )
    return {"error": f"unknown tool: {name}"}


async def _call_flight_tool(name: str, arguments: dict) -> dict:
    """Flight tools go through the MCP client (with its in-process fallback)."""
    client = get_flights_client()
    if name == "get_flight_status":
        return await client.status(
            arguments.get("flight_no", ""), arguments.get("date")
        )
    return {"delays": await client.delays(arguments.get("route"), arguments.get("date"))}


async def _call_ops_tool(name: str, arguments: dict) -> dict:
    """Booking/rebooking/tracking ops go through the ops MCP server."""
    try:
        client = get_ops_client()
        payload = await client.call_tool(name, arguments)
        if isinstance(payload, dict):
            return payload
        if payload is None:
            return {}
        return {"result": payload}
    except Exception:
        logger.warning("ops MCP call failed for %s; using in-process fallback", name, exc_info=True)
        if name == "rebook_flight":
            return _rebook(arguments)
        if name == "confirm_rebook":
            return await booking.confirm_rebook(
                arguments.get("ticket_number", ""),
                arguments.get("new_date", ""),
                arguments.get("flight_no", ""),
                email=arguments.get("email") or None,
                passenger_name=arguments.get("passenger_name") or None,
            )
        if name == "search_flights":
            return await asyncio.to_thread(
                booking.search_flights,
                arguments.get("origin", ""),
                arguments.get("destination", ""),
                arguments.get("date"),
            )
        if name == "create_booking":
            return await booking.create_booking(
                arguments.get("flight_no", ""),
                arguments.get("date"),
                passenger_name=arguments.get("passenger_name") or None,
                passenger_email=arguments.get("passenger_email") or None,
                passenger_id=arguments.get("passenger_id"),
                conversation_id=arguments.get("conversation_id"),
            )
        if name == "complete_booking_email":
            return await booking.complete_booking_email(
                arguments.get("pnr", ""),
                arguments.get("email", ""),
                passenger_name=arguments.get("passenger_name") or None,
            )
        raise


def _inject_booking_context(arguments: dict, context: dict | None) -> dict:
    """Add internal identity to create_booking without trusting the model.

    The LLM supplies the flight, date and (optionally) an email; the passenger
    id, profile email/name and conversation id come from the authenticated
    turn. The M-Pesa number is intentionally not injected: the payment card
    collects it after the card is rendered.
    """
    if not context:
        return arguments
    merged = dict(arguments)
    for key in ("passenger_id", "passenger_email",
                "passenger_name", "conversation_id"):
        if not merged.get(key) and context.get(key):
            merged[key] = context[key]
    return merged


async def call_tool(name: str, arguments: dict, *, context: dict | None = None) -> dict:
    """Execute one tool; flights via flights MCP, bookings/rebookings via ops MCP."""
    name = _canonical_name(name)
    if name == "create_booking":
        arguments = _inject_booking_context(arguments, context)
    try:
        if name in ("get_flight_status", "list_delays"):
            result = await _call_flight_tool(name, arguments)
        elif name in (
            "rebook_flight",
            "confirm_rebook",
            "search_flights",
            "create_booking",
            "issue_ticket",
            "mpesa_stk_push",
            "verify_payment",
        ):
            result = await _call_ops_tool(name, arguments)
        else:
            result = await asyncio.to_thread(_run_tool, name, arguments)
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}
    if name == "rebook_flight" and result.get("rebooked"):
        await _send_rebook_confirmation(result)
    return result


def _parse_arguments(raw: str | None) -> dict:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        logger.warning("Tool call had invalid JSON arguments: %r", raw)
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def run_tool_turn(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    llm: LLM | None = None,
    on_token=None,
    max_rounds: int = MAX_TOOL_ROUNDS,
    results: list[dict] | None = None,
    context: dict | None = None,
) -> str:
    """Offer tools, execute any the model picks, and return the final answer.

    Tools are passed on *every* call: asking the model for a final answer with
    no tools present makes gpt-oss try a tool call anyway, which Groq rejects
    with "Tool choice is none, but model called a tool". The loop stops as soon
    as the model replies with content instead of tool calls.

    `results` collects the raw tool payloads (a rebooking result carries the
    ticket, the new confirmation reference, the flight and the date). The
    caller writes them into the conversation memory, so the booking is
    remembered exactly as the tool reported it instead of being re-parsed out
    of the model's prose.
    """
    llm = llm or get_llm()
    specs = tools if tools is not None else TOOL_SPECS
    working = list(messages)

    for _ in range(max_rounds):
        message = await llm.tool_message(
            working, specs, max_tokens=settings.llm_max_answer_tokens
        )
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            content = message.get("content") or ""
            if on_token is not None and content:
                await on_token(content)
            return content

        working.append(message)
        for call in tool_calls:
            function = call.get("function", {})
            result = await call_tool(
                function.get("name", ""),
                _parse_arguments(function.get("arguments")),
                context=context,
            )
            if results is not None:
                results.append(result)
            working.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(result),
                }
            )

    # Still calling tools after the budget: ask once more and take whatever it says.
    message = await llm.tool_message(
        working, specs, max_tokens=settings.llm_max_answer_tokens
    )
    content = message.get("content") or ""
    if on_token is not None and content:
        await on_token(content)
    return content
