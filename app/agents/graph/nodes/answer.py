"""Answer-generation node and the deterministic booking fallback."""

import asyncio
import logging
import re

from app.agents import payments
from app.agents.llm import LLMError, get_llm
from app.agents.tools import (
    BOOKING_TOOL_SPECS,
    detect_booking_request,
    detect_rebook_request,
    run_tool_turn,
)
from app.config import settings
from services import booking

from ..constants import (
    BAGGAGE_INSTRUCTION,
    BAGGAGE_OPTIONS,
    BOOKING_TOOL_INSTRUCTION,
    HISTORY_TURNS_FOR_CONTEXT,
    QA_SYSTEM_PROMPT,
    REBOOK_TOOL_INSTRUCTION,
    SOFT_ESCALATION_OPTIONS,
    SOFT_OFFER_INSTRUCTION,
)
from ..state import AgentState
from .retrieval import _format_context

logger = logging.getLogger(__name__)


def _known_context_block(state: AgentState) -> str:
    """Known identity + memory slots, rendered for the QA prompt.

    Injecting these keeps the bot from re-asking for a booking reference, a
    phone number, or a travel date it was already given, and lets a logged-in
    passenger's profile/bookings be inherited without any re-entry.
    """
    facts = state.get("facts") or {}
    profile = state.get("profile") or {}
    lines: list[str] = []

    name = profile.get("name")
    if name and not profile.get("anonymous"):
        lines.append(f"Passenger name: {name}")
    if profile.get("email"):
        lines.append(f"Passenger email: {profile['email']}")
    if facts.get("ticket_number"):
        reference = str(facts["ticket_number"])
        if re.fullmatch(r"\d{13}", reference):
            lines.append(f"E-ticket number (13-digit ticket number): {reference}")
        else:
            # Most passengers call their booking reference their "ticket
            # number". Saying so here stops the bot from asking for a 13-digit
            # e-ticket number it already has the answer to.
            lines.append(
                f"Booking reference (PNR): {reference} - this is the passenger's "
                "ticket number: use it whenever they ask for their ticket "
                "number, booking reference or PNR"
            )
    if facts.get("confirmation_ref"):
        lines.append(f"Rebooking confirmation reference: {facts['confirmation_ref']}")
    if facts.get("flight_no"):
        lines.append(f"Flight: {facts['flight_no']}")
    if facts.get("new_flight_no"):
        lines.append(f"Flight after rebooking: {facts['new_flight_no']}")
    if facts.get("route"):
        lines.append(f"Route: {facts['route']}")
    if facts.get("phone"):
        lines.append(f"Contact phone already given: {facts['phone']}")
    if facts.get("email") and facts.get("email") != profile.get("email"):
        lines.append(f"Contact email already given: {facts['email']}")
    if facts.get("new_date"):
        lines.append(f"Travel date already mentioned: {facts['new_date']}")
    if facts.get("booking_pnr"):
        route = facts.get("booking_route") or "the route on file"
        amount = facts.get("booking_amount")
        fare = (
            f" for {booking.format_money(amount)}" if amount is not None else ""
        )
        lines.append(
            f"Current booking reference (PNR): {facts['booking_pnr']} - "
            f"{facts.get('booking_flight') or 'flight'} {route} on "
            f"{facts.get('booking_date') or 'the date on file'}{fare} "
            f"({facts.get('booking_status') or 'held'})"
        )
    if facts.get("eticket_number"):
        lines.append(f"E-ticket number issued: {facts['eticket_number']}")
    if facts.get("payment_id"):
        lines.append(
            "An in-chat M-Pesa payment card is attached to this booking; the "
            "number is entered there and must never be requested in prose"
        )
    if facts.get("tracking_ref"):
        lines.append(f"Baggage tracking reference: {facts['tracking_ref']}")
    if facts.get("human_declined"):
        lines.append(
            "The passenger declined a human hand-off — answer without offering one"
        )
    for existing_booking in profile.get("bookings") or []:
        route = (
            f"{existing_booking.get('origin') or '?'}-"
            f"{existing_booking.get('destination') or '?'}"
        )
        lines.append(
            f"Upcoming booking {existing_booking.get('pnr')}: "
            f"{existing_booking.get('flight_no')} {route} on "
            f"{existing_booking.get('date')} ({existing_booking.get('status')})"
        )

    if not lines:
        return ""
    return (
        "\n\nKNOWN PASSENGER CONTEXT (facts the passenger already gave you - "
        "answer questions about their own booking straight from these, never "
        "ask for them again, and never say the information is unavailable):\n"
        + "\n".join(f"- {line}" for line in lines)
    )


async def answer(state: AgentState, config=None) -> dict:
    # Complete a pending email capture: the passenger is replying with their
    # email address after a booking was held. This runs before intent routing
    # so the reply is never mis-classified as a new booking request.
    facts = state.get("facts") or {}
    if facts.get("needs_email"):
        email_match = _EMAIL_RE.search(state.get("query", "") or "")
        if not email_match:
            return {"answer": _reask(
                state,
                "What's your email address? For example name@example.com.",
            ), "chunks": []}
        email = email_match.group(0).lower()
        pending_email = facts.get("pending_email_booking") or {}
        pnr = pending_email.get("pnr")
        if pnr:
            updated = await booking.set_booking_email(pnr, email)
            if updated.get("updated"):
                ticket_result = await booking.issue_ticket(pnr, email=email)
                facts.pop("needs_email", None)
                facts.pop("pending_email_booking", None)
                eticket = ticket_result.get("eticket_number") or ""
                eticket_display = f"E-ticket {eticket}" if eticket else "E-ticket issued"
                email_sent = ticket_result.get("email_sent", False)
                email_note = ticket_result.get("email_note", "")
                if not ticket_result.get("ticketed"):
                    return {"answer": (
                        f"Thank you for booking the flight! Your email ({email}) "
                        f"has been saved to booking {pnr}.\n"
                        f"{email_note or 'Your e-ticket will be issued and emailed once the M-Pesa payment completes.'}"
                    ), "chunks": []}
                origin = updated.get("origin") or ""
                destination = updated.get("destination") or ""
                route_label = f"{origin} → {destination}" if origin and destination else ""
                date_label = booking.human_date(updated.get("date") or "")
                itinerary = (
                    "| Flight | Route | Date | Departs | Fare |\n"
                    "|--------|-------|------|---------|------|\n"
                    f"| {updated.get('flight_no') or ''} | {route_label} | "
                    f"{date_label} | {updated.get('depart_time') or ''} | "
                    f"{booking.format_money(float(updated.get('amount') or 0), updated.get('currency') or 'KES')} |\n"
                )
                try:
                    pid = pending_email.get("payment_id")
                    if pid:
                        await payments.broadcast_payment_update(pid)
                except Exception:
                    pass
                return {"answer": (
                    f"Thank you for booking the flight! Your email ({email}) has been "
                    f"saved to booking {pnr}. {eticket_display} has been issued.\n\n"
                    f"Here's your booking:\n\n{itinerary}\n"
                    f"Booking reference: {pnr}.\n"
                    f"{'Confirmation email sent to ' + email + '.' if email_sent else email_note}"
                ), "chunks": []}
            return {"answer": updated.get("error", "I couldn't save your email."), "chunks": []}
        return {"answer": "I don't have a booking on file to attach your email to.", "chunks": []}

    context, _ = _format_context(state.get("chunks", []))
    system = QA_SYSTEM_PROMPT.format(context=context or "(no context)")
    if not context:
        # Nothing in the knowledge base covers this. Give the model the
        # official KQ channels so it can offer real next steps instead of
        # shrugging or handing the passenger off.
        system += (
            "\n\nOFFICIAL KQ CHANNELS (use these as the concrete next steps "
            "when the context does not cover the question):\n"
            f"- Self-service help centre: {settings.kq_help_url}\n"
            f"- Manage your booking: {settings.kq_manage_booking_url}\n"
            f"- Customer care: {settings.kq_contact_phone} "
            f"({settings.kq_contact_hours})"
        )
    flight_context = state.get("flight_context")
    if flight_context:
        system += (
            "\n\nLive flight information from the flight service (mock data):\n"
            + flight_context
        )
    rebook_request = detect_rebook_request(state.get("query", ""), state.get("history"))
    booking_request = detect_booking_request(state.get("query", ""), state.get("history"))
    baggage_context = state.get("baggage_context")
    if baggage_context:
        system += "\n\n" + BAGGAGE_INSTRUCTION.format(guidance=baggage_context)
    system += _known_context_block(state)
    if state.get("offer_escalation"):
        system += "\n\n" + SOFT_OFFER_INSTRUCTION
    # The tool instruction is appended only for the tool attempt, so the
    # no-tools fallback below answers without re-offering tools.
    tool_instruction = ""
    if rebook_request:
        tool_instruction = "\n\n" + REBOOK_TOOL_INSTRUCTION.format(
            self_service_url=settings.self_service_rebooking_url
        )
    elif booking_request:
        tool_instruction = "\n\n" + BOOKING_TOOL_INSTRUCTION
    messages = [
        {
            "role": "system",
            "content": system + tool_instruction,
        },
        *state.get("history", [])[-HISTORY_TURNS_FOR_CONTEXT:],
        {"role": "user", "content": state.get("query", "")},
    ]

    on_token = None
    if config:
        on_token = config.get("configurable", {}).get("on_token")

    tool_results: list[dict] = []
    # Tool asks run through LLM function calling: the model collects the
    # missing slot, calls the tool, then answers from the result. If that
    # fails, fall back to a plain answer; bookings additionally have a
    # deterministic search/create path that needs no model at all.
    if rebook_request or booking_request:
        # For new bookings, try the deterministic fallback first: it reliably
        # collects route/date/phone and creates the booking without depending
        # on the LLM to pick the right tools. Rebooking still goes through the
        # LLM tool turn because it needs multi-turn coordination.
        if booking_request:
            text = await _booking_fallback(state, _booking_context(state), tool_results)
            if text:
                return _with_offer(state, text, baggage_context, tool_results)

        specs = BOOKING_TOOL_SPECS if booking_request and not rebook_request else None
        try:
            text = await run_tool_turn(
                messages,
                tools=specs,
                on_token=on_token,
                results=tool_results,
                context=_booking_context(state),
            )
        except LLMError:
            logger.warning("Booking/rebooking tool turn failed; answering without tools")
            text = ""
        if not text and booking_request:
            text = await _booking_fallback(state, _booking_context(state), tool_results)
        if text:
            return _with_offer(state, text, baggage_context, tool_results)
        messages[0] = {"role": "system", "content": system}

    llm = get_llm()
    try:
        if on_token is not None:
            parts: list[str] = []
            async for token in llm.stream(messages, max_tokens=settings.llm_max_answer_tokens):
                parts.append(token)
                await on_token(token)
            text = "".join(parts)
        else:
            text = await llm.complete(messages, max_tokens=settings.llm_max_answer_tokens)
    except LLMError as exc:
        logger.error("QA generation failed: %s", exc)
        text = "I'm sorry, I couldn't process that just now. Please try again."

    return _with_offer(state, text, baggage_context, tool_results)


def _booking_context(state: AgentState) -> dict:
    """Identity the backend injects into create_booking.

    The model only supplies the flight and date; the passenger id, profile
    email/name and conversation id come from the authenticated turn. Phone is
    collected in conversation (not asked for as a name) and passed through
    context when already known.
    """
    profile = state.get("profile") or {}
    facts = state.get("facts") or {}
    return {
        "passenger_id": state.get("passenger_id"),
        "passenger_name": profile.get("name") or facts.get("passenger_name"),
        "passenger_email": profile.get("email") or facts.get("email"),
        "passenger_phone": facts.get("phone"),
        "conversation_id": state.get("conversation_id"),
    }


_FLIGHT_NO_RE = re.compile(r"\bKQ\s?\d{1,4}\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

_BOOKING_APOLOGY = "Sorry, I didn't get that clearly. "


def _reask(state: dict, question: str) -> str:
    """Return a booking question, apologising when it is a repeat.

    The previous question is remembered in the conversation facts. Asking the
    same question again means the passenger's last reply was not understood,
    so the re-ask opens with an apology instead of sounding stuck.
    """
    facts = state.get("facts")
    if facts is None:
        facts = {}
        state["facts"] = facts
    if facts.get("last_booking_question") == question:
        return _BOOKING_APOLOGY + question
    facts["last_booking_question"] = question
    return question


async def _booking_fallback(
    state: AgentState,
    context: dict,
    results: list[dict] | None = None,
) -> str:
    """Deterministic booking path for when the model/tool turn is unavailable.

    Scans the current turn and recent history for a flight number, route and
    date. With a flight + date + phone it creates the held booking; with a
    flight + date but no phone it asks for the phone number; with a route +
    date it lists the schedule; otherwise it asks for the missing slot.
    Nothing is invented.
    """
    query = state.get("query", "")
    facts = state.get("facts") or {}

    query = state.get("query", "")
    # Only the passenger's own words may fill the flight/route/date slots.
    # Bot replies must never be re-parsed as the passenger's answer: the
    # booking questions themselves name example routes ("For example Nairobi
    # to Mombasa"), and feeding those back silently booked the wrong route.
    passenger_texts = [
        m.get("content", "")
        for m in reversed(state.get("history") or [])
        if m.get("role") == "user"
    ]
    texts = [query, *passenger_texts]
    flight_no = None
    route = None
    travel_date = None
    # Track whether the flight number came from the current turn (a passenger
    # choosing a flight from the table) vs. stale history.
    flight_no_in_query = False
    for i, text in enumerate(texts):
        if flight_no is None:
            match = _FLIGHT_NO_RE.search(text or "")
            if match:
                flight_no = match.group(0).replace(" ", "").upper()
                flight_no_in_query = (i == 0)
        if route is None:
            route = booking.parse_route(text or "")
        if travel_date is None:
            travel_date = booking.parse_travel_date(text or "")
        if route and travel_date:
            break

    passenger_phone = context.get("passenger_phone")
    facts_phone = (state.get("facts") or {}).get("phone")
    # A travel date may have been captured in conversation facts (e.g. "tomorrow"
    # from an earlier turn) even when the current scan doesn't find one.
    facts_date = None
    if (state.get("facts") or {}).get("new_date"):
        facts_date = booking.parse_travel_date((state.get("facts") or {}).get("new_date"))

    if travel_date is None and facts_date is not None:
        travel_date = facts_date

    # Complete a pending booking when the passenger supplies their phone number.
    pending = (state.get("facts") or {}).get("pending_booking")
    if pending:
        phone_match = re.search(r"(?:\+?254|0)\s?[17]\d{2}[\s-]?\d{3}[\s-]?\d{3}\b", query or "")
        if phone_match:
            phone = re.sub(r"[\s-]", "", phone_match.group(0))
            travel_date = booking.parse_travel_date(pending["travel_date"])
            flight_no = pending["flight_no"]
            result = await booking.create_booking(
                flight_no,
                pending["travel_date"],
                passenger_phone=phone,
                passenger_email=context.get("passenger_email"),
                passenger_id=context.get("passenger_id"),
                conversation_id=context.get("conversation_id"),
            )
            if results is not None:
                results.append(result)
            if not result.get("booked"):
                return result.get("error") or "I couldn't create that booking."
            facts = state.get("facts") or {}
            facts.pop("pending_booking", None)
            row = booking._schedule_row(flight_no)
            if row is None:
                return f"I couldn't find flight {flight_no}."
            _, origin, destination, depart, _arrive = row
            amount = booking._fare(flight_no, origin, destination, travel_date)
            # Trigger the M-Pesa STK push so the passenger receives the prompt
            # without having to open the payment card UI.
            payment_id = result.get("payment_id")
            if payment_id:
                try:
                    await payments.start_payment(str(payment_id), phone)
                except Exception as exc:
                    return (
                        f"I've held {flight_no} ({origin}-{destination}) on "
                        f"**{booking.human_date(travel_date.isoformat())}** at {depart} for "
                        f"{booking.format_money(amount)}. Booking reference: "
                        f"{result.get('pnr')}.\n"
                        f"M-Pesa prompt could not be sent: {exc}\n"
                        "Enter your M-Pesa number in the payment card to retry."
                    )
                # Ensure the frontend sees the updated payment state immediately.
                await payments.broadcast_payment_update(str(payment_id))
            return (
                f"I've held {flight_no} ({origin}-{destination}) on "
                f"**{booking.human_date(travel_date.isoformat())}** at {depart} for "
                f"{booking.format_money(amount)}. Booking reference: "
                f"{result.get('pnr')}.\n"
                "Your M-Pesa payment card is below — the prompt should arrive "
                "shortly. Enter your PIN on your phone to complete payment."
            )
        # The reply was not a phone number: apologise and ask again.
        return _reask(
            state,
            "What's your phone number for the M-Pesa payment? (e.g. 07xx xxx xxx)",
        )

    # With a route + date, always search flights first (even if a stale flight
    # number lingers in history). Only create a booking directly when the
    # passenger explicitly chose a flight in *this* turn.
    if flight_no and travel_date and flight_no_in_query:
        # We have a flight and date. If we don't have a phone yet, ask for it
        # before creating the booking (the user wants phone in chat, not name).
        if not passenger_phone and not facts_phone:
            # Remember the pending booking so the phone reply can complete it.
            pending = {
                "flight_no": flight_no,
                "travel_date": travel_date.isoformat(),
            }
            facts = state.get("facts") or {}
            facts["pending_booking"] = pending
            return _reask(
                state,
                f"I've found {flight_no} for **{booking.human_date(travel_date.isoformat())}**. "
                "What's your phone number for the M-Pesa payment? (e.g. 07xx xxx xxx)",
            )
        phone = passenger_phone or facts_phone
        # Fallback path: create the held booking directly so the payment card
        # is still rendered.
        result = await booking.create_booking(
            flight_no,
            travel_date.isoformat(),
            passenger_phone=phone,
            passenger_email=context.get("passenger_email"),
            passenger_id=context.get("passenger_id"),
            conversation_id=context.get("conversation_id"),
        )
        if results is not None:
            results.append(result)
        if not result.get("booked"):
            return result.get("error") or "I couldn't create that booking."
        row = booking._schedule_row(flight_no)
        if row is None:
            return f"I couldn't find flight {flight_no}."
        _, origin, destination, depart, _arrive = row
        amount = booking._fare(flight_no, origin, destination, travel_date)
        # Trigger the M-Pesa STK push so the passenger receives the prompt
        # without having to open the payment card UI.
        payment_id = result.get("payment_id")
        if payment_id:
            try:
                await payments.start_payment(str(payment_id), phone)
            except Exception as exc:
                return (
                    f"I've held {flight_no} ({origin}-{destination}) on "
                    f"**{booking.human_date(travel_date.isoformat())}** at {depart} for "
                    f"{booking.format_money(amount)}. Booking reference: "
                    f"{result.get('pnr')}.\n"
                    f"M-Pesa prompt could not be sent: {exc}\n"
                    "Enter your M-Pesa number in the payment card to retry."
                )
            await payments.broadcast_payment_update(str(payment_id))
        return (
            f"I've held {flight_no} ({origin}-{destination}) on "
            f"**{booking.human_date(travel_date.isoformat())}** at {depart} for "
            f"{booking.format_money(amount)}. Booking reference: "
            f"{result.get('pnr')}.\n"
            "Your M-Pesa payment card is below — the prompt should arrive "
            "shortly. Enter your PIN on your phone to complete payment."
        )

    if route and travel_date:
        result = await asyncio.to_thread(
            booking.search_flights, route[0], route[1], travel_date.isoformat()
        )
        # Resolve labels:
        origin_label = booking.airport_label(route[0])
        destination_label = booking.airport_label(route[1])
        if not result.get("available"):
            return (
                f"Unfortunately there are no available flights from {origin_label} "
                f"to {destination_label} on **{booking.human_date(travel_date.isoformat())}**.\n\n"
                f"| Flight | Departs | Arrives | Fare |\n"
                f"|--------|---------|---------|------|\n"
                f"| — | — | — | — |\n\n"
                "Try another date or route."
            )
        lines = [
            f"Flights {origin_label} to {destination_label} on "
            f"**{booking.human_date(result['date'])}**:",
            "",
            "| Flight | Departs | Arrives | Fare | Seats |",
            "|--------|---------|---------|------|-------|",
        ]
        for flight in result.get("flights", [])[:10]:
            lines.append(
                f"| {flight['flight_no']} | {flight['depart_time']} | "
                f"{flight['arrival_time']} | {booking.format_money(flight['fare'], flight.get('currency', 'KES'))} | "
                f"{flight.get('seats_left', 9)} |"
            )
        lines.append("")
        lines.append("Tell me which flight you'd like and I'll hold it for you.")
        return "\n".join(lines)

    if route and not travel_date:
        return _reask(
            state,
            f"Happy to book {booking.airport_label(route[0])} to "
            f"{booking.airport_label(route[1])}. Which date would you like to travel?",
        )
    if travel_date and not route:
        return _reask(
            state,
            "Which route would you like to fly? For example Nairobi to Mombasa.",
        )
    return _reask(
        state,
        "I can book that for you. Where are you flying from and to, and on "
        "which date?",
    )


def _with_offer(
    state: AgentState,
    text: str,
    baggage_context: str | None,
    tool_results: list[dict],
) -> dict:
    """Attach the quick replies that let the passenger pull in a human.

    Baggage guidance wins over a generic soft trigger: it already carries its
    own "Connect me to a human" option.
    """
    result: dict = {"answer": text}
    if tool_results:
        result["tool_results"] = tool_results
    if baggage_context:
        result["escalation_offer"] = {
            "reason": "lost or delayed baggage",
            "options": [dict(option) for option in BAGGAGE_OPTIONS],
        }
    elif state.get("offer_escalation"):
        result["escalation_offer"] = {
            "reason": state.get("escalation_reason", ""),
            "options": [dict(option) for option in SOFT_ESCALATION_OPTIONS],
        }
    return result
