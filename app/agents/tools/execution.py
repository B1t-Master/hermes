"""Tool dispatch and the model/tool loop."""

from __future__ import annotations

import asyncio
import json
import logging

from app.agents.llm import LLM, get_llm
from app.agents.mcp_client import get_flights_client, get_ops_client
from app.config import settings
from services import booking, rebook

from .email import _send_rebook_confirmation
from .specs import TOOL_SPECS

logger = logging.getLogger(__name__)

# Cap on tool-calling rounds before we ask once more for a plain answer.
# Booking needs search -> create -> answer, hence three rounds.
MAX_TOOL_ROUNDS = 3


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
