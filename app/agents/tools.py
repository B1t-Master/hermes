"""LLM function calling for the hermes agent.

Tool schemas are OpenAI-compatible. Implementations are the same mock services
the MCP server exposes (`services/flight_delays.py`, `services/rebook.py`), so
the bot and external MCP clients share one source of truth.

`run_tool_turn` drives the model/tool loop: the model may call tools, results
are fed back as `tool` messages, and the loop returns once the model replies
with content instead of another tool call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from sqlalchemy import func, select

from app.agents.llm import LLM, get_llm
from app.config import settings
from app.db import SessionLocal
from app.models import RebookEmailMap
from services import email_real, flight_delays, rebook

logger = logging.getLogger(__name__)

# Cap on tool-calling rounds before we ask once more for a plain answer.
MAX_TOOL_ROUNDS = 2

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
    on the plain path.
    """
    if _has_rebook_intent(query):
        return True
    if not history:
        return False
    if not (_looks_like_ticket(query) or _looks_like_date(query)):
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
                "Rebook a passenger onto an available flight for a new travel "
                "date. Requires the passenger's booking reference or e-ticket "
                "number AND the new travel date (natural language like "
                "'tomorrow' or 'tomorrow afternoon' is understood); ask for "
                "whichever is missing instead of calling this tool. The new "
                "flight is chosen automatically. Emails a confirmation only "
                "when the booking reference has an email address on file; the "
                "result's `email_sent` field says whether one was sent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {
                        "type": "string",
                        "description": "Optional original flight number, when known.",
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
                            "YYYY-MM-DD. Ask for it if not provided."
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
                "required": ["ticket_number"],
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
    """Destination email for a ticket, or None when it is not on file."""
    ticket = (ticket_number or "").strip()
    if not ticket:
        return None
    async with SessionLocal() as db:
        row = (
            await db.execute(
                select(RebookEmailMap).where(
                    func.upper(RebookEmailMap.ticket_number) == ticket.upper()
                )
            )
        ).scalar_one_or_none()
    return row.email if row else None


async def _send_rebook_confirmation(result: dict) -> None:
    """Email the confirmation when the ticket has an address on file.

    Only tickets present in ``rebook_email_map`` get a confirmation; an
    unknown ticket is reported back (no email) rather than defaulted. Delivery
    problems are swallowed — the rebooking already succeeded, and the
    passenger's answer must not fail because of mail.
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
    if name == "get_flight_status":
        return flight_delays.get_flight_status(
            arguments.get("flight_no", ""), arguments.get("date")
        )
    if name == "list_delays":
        return {
            "delays": flight_delays.list_delays(
                arguments.get("route"), arguments.get("date")
            )
        }
    if name == "rebook_flight":
        return _rebook(arguments)
    return {"error": f"unknown tool: {name}"}


async def call_tool(name: str, arguments: dict) -> dict:
    """Execute one tool off the event loop (the mock services are sync/SQLite)."""
    name = _canonical_name(name)
    try:
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
) -> str:
    """Offer tools, execute any the model picks, and return the final answer.

    Tools are passed on *every* call: asking the model for a final answer with
    no tools present makes gpt-oss try a tool call anyway, which Groq rejects
    with "Tool choice is none, but model called a tool". The loop stops as soon
    as the model replies with content instead of tool calls.
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
                function.get("name", ""), _parse_arguments(function.get("arguments"))
            )
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
