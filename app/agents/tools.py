"""LLM function calling for the hermes agent.

Tool schemas are OpenAI-compatible. Implementations are the same mock services
the MCP server exposes (`services/flight_delays.py`, `services/rebook.py`), so
the bot and external MCP clients share one source of truth.

`run_tool_turn` drives the model/tool loop: the model may call tools, results
are fed back as `tool` messages, and the final answer streams via `on_token`
so the tool path keeps the same UX as the plain answer path.
"""

from __future__ import annotations

import asyncio
import json
import logging

from app.agents.llm import LLM, get_llm
from services import flight_delays, rebook

logger = logging.getLogger(__name__)

# Cap on tool-calling rounds before we ask once more for a plain answer.
MAX_TOOL_ROUNDS = 2

# Narrow phrases that mark a genuine "move me to another flight" request. Used
# to decide whether to offer the rebook tool at all; the model still chooses
# the tool and its arguments.
REBOOK_KEYWORDS = (
    "rebook",
    "re-book",
    "book me",
    "change my flight",
    "change the flight",
    "move me to",
    "move my flight",
    "reschedule my flight",
    "another flight",
    "alternative flight",
    "next flight",
    "different flight",
)


def detect_rebook_request(query: str) -> bool:
    lower = query.lower()
    return any(keyword in lower for keyword in REBOOK_KEYWORDS)


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
                "day). Returns a confirmation reference and the new flight details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {
                        "type": "string",
                        "description": 'The disrupted flight number, such as "KQ100".',
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
                "required": ["flight_no"],
            },
        },
    },
]


def _canonical_name(name: str) -> str:
    """Drop any namespace prefix gpt-oss invents (e.g. ``repo_browser.rebook_flight``)."""
    return name.rsplit(".", 1)[-1].strip()


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
        return rebook.rebook_flight(
            arguments.get("flight_no", ""),
            arguments.get("new_date"),
            arguments.get("passenger_name"),
            arguments.get("reason"),
        )
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
