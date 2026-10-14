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

from app.agents.llm import LLM, get_llm
from app.config import settings
from services import email, flight_delays, rebook

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
_TICKET_HINT = re.compile(r"\b(ticket|booking|reference|pnr|e-?ticket)\b", re.IGNORECASE)
_TICKET_NUMBER = re.compile(r"\b\d{6,14}\b")


def _has_rebook_intent(text: str) -> bool:
    return bool(_REBOOK_RE.search(text))


def _looks_like_ticket(text: str) -> bool:
    return bool(_TICKET_HINT.search(text) or _TICKET_NUMBER.search(text))


def detect_rebook_request(query: str, history: list[dict] | None = None) -> bool:
    """Whether to offer the rebook tools on this turn.

    True for a direct rebook ask, or when the passenger is answering our
    request for a booking reference right after a rebook ask (so the multi-turn
    flow keeps the tools available). Unrelated follow-ups stay on the plain
    path.
    """
    if _has_rebook_intent(query):
        return True
    if not (history and _looks_like_ticket(query)):
        return False
    recent_user = [m.get("content", "") for m in history[-4:] if m.get("role") == "user"]
    return any(_has_rebook_intent(text) for text in recent_user)


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
                "Rebook a passenger whose flight was cancelled or rescheduled "
                "onto the same flight number on a new date (default: the next "
                "day). Requires the passenger's booking reference or e-ticket "
                "number. Emails a confirmation. Returns the new flight details "
                "and a confirmation reference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {
                        "type": "string",
                        "description": 'The disrupted flight number, such as "KQ100".',
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
                        "description": "Preferred new travel date YYYY-MM-DD; defaults to the next day.",
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
                "required": ["flight_no", "ticket_number"],
            },
        },
    },
]


def _canonical_name(name: str) -> str:
    """Drop any namespace prefix gpt-oss invents (e.g. ``repo_browser.rebook_flight``)."""
    return name.rsplit(".", 1)[-1].strip()


def _rebook(arguments: dict) -> dict:
    result = rebook.rebook_flight(
        arguments.get("flight_no", ""),
        ticket_number=arguments.get("ticket_number"),
        new_date=arguments.get("new_date"),
        passenger_name=arguments.get("passenger_name"),
        reason=arguments.get("reason"),
    )
    if result.get("rebooked"):
        subject, body = rebook.confirmation_email(result)
        record = email.send_email(
            settings.rebook_confirmation_email,
            subject,
            body,
            outbox_dir=settings.email_outbox_dir,
        )
        result["email_sent"] = True
        result["email_to"] = settings.rebook_confirmation_email
        result["email_sent_at"] = record["sent_at"]
    return result


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
    try:
        return await asyncio.to_thread(_run_tool, name, arguments)
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}


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
        message = await llm.tool_message(working, specs)
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
    message = await llm.tool_message(working, specs)
    content = message.get("content") or ""
    if on_token is not None and content:
        await on_token(content)
    return content
