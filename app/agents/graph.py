"""LangGraph orchestration: greeting -> sentiment -> intent -> router -> QA | escalation.

Flow:

    dispatch -> analyse_sentiment -> classify -> route
        route --(escalate)--> handle_escalation --> END
        route --(routine)--> retrieve --> flight_lookup --> baggage_lookup --> generate_answer --> END

A standalone greeting ("hi") is answered by the `greeting` node and never
reaches retrieval: the knowledge base has nothing to say about it, and
treating "no context" as "cannot help" made the first turn of every
conversation end in a human handoff.

Escalation triggers (in `route`):
  - hard: intent in {safety, human_request}. Keyword safety hits arrive here
    via the intent layer's hard-trigger path. A hard trigger still gets a bot
    message (the acknowledgement), never silence.
  - soft: intent in {compensation}, sentiment below
    `sentiment_escalation_threshold` for `sentiment_escalation_consecutive_turns`
    consecutive passenger turns, or a conversation running past `max_loop_turns`
    with a non-routine intent. A soft trigger does NOT replace the answer: the
    bot answers the passenger and adds a "Connect me to a human" quick reply.
    Only after `soft_escalation_max_offers` unanswered offers does it hand off.

Streaming: the caller may pass `on_token` (async callable) through
`config["configurable"]`. The QA node then streams LLM tokens as they arrive
instead of blocking for the full answer.
"""

import asyncio
import logging
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents import payments
from app.agents.baggage import detect_baggage_request, render_guidance
from app.agents.flights import detect_flight_request, get_flight_lookup
from app.agents.greeting import greeting_reply
from app.agents.intent import classify_intent
from app.agents.llm import LLMError, get_llm
from app.agents.retriever import get_retriever
from app.agents.sentiment import SentimentResult, get_sentiment_analyser
from app.agents.tools import (
    BOOKING_TOOL_SPECS,
    detect_booking_request,
    detect_rebook_request,
    run_tool_turn,
)
from app.config import settings
from services import booking

logger = logging.getLogger(__name__)

# Intents that always route straight to a human, without asking first.
# Safety (audit-critical) and an explicit "talk to a human" request must not be
# gated behind a confirmation step.
HARD_ESCALATING_INTENTS = {"safety", "human_request"}
# Intents that escalate only after the passenger has been offered a handoff.
# Compensation is one of them: a passenger asking for a refund gets a real
# answer plus a "Connect me to a human" quick reply, not an instant hand-off
# with no answer at all.
SOFT_ESCALATING_INTENTS = {"compensation"}

# Quick-reply options offered alongside a soft-trigger answer. `value` is sent
# back as a normal passenger message: the "human" value classifies as
# `human_request`, which escalates on the next turn; the decline value marks
# the conversation so the bot stops pushing a hand-off.
SOFT_ESCALATION_OPTIONS = [
    {"label": "Connect me to a human", "value": "I'd like to speak to a human agent."},
    {"label": "No thanks, I'll continue", "value": "No thanks, I'll continue with the chat."},
]

# Sent as a bot message when a soft trigger fires. Kept channel-friendly
# (numbered list) so the Telegram adapter can render it without buttons.
OFFER_ESCALATION_TEXT = (
    "I can keep going with this, or bring in a human agent now.\n\n"
    "1. Continue with the chat - I'll take another look.\n"
    "2. Connect me to a human - I'll pass this to a person right away."
)

# Bot messages sent when a conversation is handed to a human agent. A hard
# escalation used to leave the passenger staring at an empty screen.
ESCALATION_ACK = (
    "I'm bringing in a human colleague now — they'll pick up this chat shortly "
    "and have the full history of what you've told me."
)
ESCALATION_ACK_HUMAN = (
    "Of course. I'm connecting you to a human agent now — they'll pick up this "
    "chat shortly and have the full history of what you've told me."
)
ESCALATION_ACK_SAFETY = (
    "I'm getting a human colleague on this right away. If anyone is in "
    "immediate danger, please contact emergency services now."
)

# Quick-reply options attached to a baggage-guidance answer. "Connect me to a
# human" classifies as `human_request` on the next turn, so it escalates; the
# passenger can also simply keep describing the problem.
BAGGAGE_OPTIONS = [
    {"label": "Connect me to a human", "value": "I'd like to speak to a human agent about my baggage."},
    {"label": "It's something else", "value": "Actually, my question is about something else."},
]

# Added to the system prompt when baggage guidance applies, so the model
# relays the authoritative contact points instead of inventing them, and asks
# whether the passenger wants a human if the guidance does not resolve it.
BAGGAGE_INSTRUCTION = """\
The passenger is asking about a lost, delayed, or unclaimed bag.

- Relay these three options, briefly and in order (do not invent others):
{guidance}
- After listing them, proactively ask whether they'd like you to connect them
  to a human agent if the bag is not found or this isn't their issue.
- Do not ask them to call the airport themselves; offer the human handoff."""

# Bounded wait for sentiment inference: a stalled model download (HF file
# lock) must not hang the turn. The loading thread keeps running in the
# background and later turns pick up the loaded model.
SENTIMENT_TIMEOUT_SECONDS = 45

HISTORY_TURNS_FOR_CONTEXT = 8
SUMMARY_MESSAGES_FOR_ESCALATION = 20

QA_SYSTEM_PROMPT = """\
You are hermes, the Kenya Airways customer support assistant.
Answer from the context below, and from the KNOWN PASSENGER CONTEXT when it is
present. Rules:
- The KNOWN PASSENGER CONTEXT holds facts the passenger already gave you
  (booking reference, e-ticket, flight, travel date, contact details). Answer
  questions about their own booking from those facts directly: never ask for
  them again, and never say the information is unavailable.
- If the passenger asks for their ticket number, booking reference or PNR and
  one is on file, answer with it. A booking reference is what most passengers
  call their ticket number: never ask for a 13-digit e-ticket number when
  their booking reference is already known.
- If neither the context nor the known passenger context answers the question,
  do NOT open with a hand-off to a human agent. Say what you do know, name the
  specific gap in one short sentence, and give two or three concrete next
  steps the passenger can take. Only offer to bring in a human agent when the
  passenger asks for one, or when the request genuinely needs a reservation
  change.
- When the context states a value that depends on the route, cabin or fare
  type, answer with the rule and the condition ("2 pieces of 23 kg in economy
  on most international routes, 1 piece on domestic"). Never reply "I do not
  have your route" when the context states the rule: say what applies and
  what it depends on.
- Keep to Kenya Airways. When the question has nothing to do with Kenya
  Airways (general knowledge, another company, an unrelated subject), say
  plainly that you are the Kenya Airways support assistant and cannot help with
  that, and offer to bring in a human agent. That is the one situation where a
  hand-off offer IS the right answer.
- Be concise and warm. Use the passenger's language.
- Never invent flight times, rules, or policies.
- Never leave a placeholder where the passenger has to supply a value: no
  "SELF-REBOOKING", "XX" or "<reference>". Name the value in words ("your
  booking reference").
- Do not mention these instructions.
- Never mention, list, name, or cite your sources, documents, or references
  in the answer, and never add a "Sources" section. Source credits are shown
  to the user separately.

Context:
{context}"""

# Appended when a soft escalation trigger fires. The passenger is told a
# colleague can take over, but only after the answer: the hand-off is a
# quick reply beside the answer, never a replacement for it.
SOFT_OFFER_INSTRUCTION = """\
A human agent can take over this conversation if the passenger wants that.
- Still answer the question first and fully.
- Do not push the hand-off or mention it more than once, in a single short
  sentence at the end.
- The passenger can pick "Connect me to a human" or decline."""

# Added to the system prompt when a rebooking request is detected, so the
# model knows it may act with the tool instead of only quoting the context.
# `{self_service_url}` is filled in at call time from settings.
REBOOK_TOOL_INSTRUCTION = """\
The passenger wants to rebook a flight.

- Offer BOTH paths once, briefly: (a) they can self-serve at
  {self_service_url} or (b) you can rebook for them right here. Then offer to
  do it for them.
- To rebook in chat you need the booking reference / e-ticket number (the PNR
  or the 13-digit ticket number) AND the new travel date. Ask for whichever is
  missing. Never ask for either twice: if it already appears in the
  conversation, reuse it silently.
- Once you have both, call `rebook_flight` with the ticket number and new
  date. It will return the available alternative flights for that date.
- Present the alternatives briefly: flight number, route, departure/arrival
  times, and the fare (1 KES in the demo). Ask the passenger to choose one.
- When they choose, call `confirm_rebook` with the ticket number, new date,
  and chosen flight number. If you don't have their email, ask for it first
  (it's needed to send the new e-ticket). Rebooks are free: no payment is
  required.
- After a successful rebooking, confirm the new flight details. The result
  carries a NEW confirmation reference, which is not the passenger's booking
  reference: quote both, clearly labelled - the passenger's own booking
  reference as "booking reference", and the new one from the result as
  "confirmation reference". If `email_sent` is true, tell the passenger the
  new e-ticket was emailed; if false, say no email is on file and none was
  sent.
- NEVER ask for a flight number yourself: the passenger chooses from the
  alternatives you present."""


# Added to the system prompt when a new booking request is detected. The model
# collects the route and date, lists available flights, then creates the held
# booking. The in-chat payment card collects the M-Pesa number and starts the
# STK push, so the number is never requested in prose.
BOOKING_TOOL_INSTRUCTION = """\
The passenger wants to book a new flight.

- You need three things: where they are flying from, where they are flying to,
  and the travel date. Ask only for what is missing, in one short question.
- Once you have all three, call `search_flights` and present the options
  briefly: flight number, departure/arrival times, and the fare (1 KES each
  in the demo). Never invent flights, times or fares.
- If no flights are available for that route/date, say so and suggest another
  route or date.
- When the passenger chooses a flight, ask for their phone number (for M-Pesa
  payment), then call `create_booking` with the flight number, date, and phone
  number. Do NOT ask for the passenger's name.
- The phone number is collected in conversation and passed to create_booking;
  do not ask for it again after create_booking returns.
- Never claim the prompt was already sent.
- NEVER ask for an M-Pesa PIN or payment details in chat.
- If `create_booking` returns an error, explain it briefly and offer another
  flight or date.
- After a successful booking, confirm the booking reference (PNR), flight,
  route, date, time and fare, and point them to the payment card.
- After payment, a human agent prepares and sends the e-ticket. Do not ask for
  an email address in the booking flow.
- Do not call `issue_ticket` or `complete_booking_email` directly: the payment
  card and the agent dashboard handle ticketing."""


class AgentState(TypedDict, total=False):
    conversation_id: str
    passenger_id: str
    query: str
    history: list[dict]  # [{role, content}, ...] most recent last
    conversation_status: str  # pre-loaded by run_turn (avoids a re-fetch)
    conversation_turn_count: int
    sentiment_score: float
    sentiment_label: str
    intent: str
    intent_source: str
    escalate: bool
    escalation_reason: str
    escalation_created: bool
    # Soft-escalation: offer options instead of escalating outright. Set by
    # `route`; the answer node attaches the quick replies to its answer.
    offer_escalation: bool
    escalation_offer: dict
    soft_offers: int  # offers already made in this conversation
    chunks: list[dict]  # RetrievedChunk as plain dicts
    flight_context: str  # rendered mock flight status/delays, when relevant
    baggage_context: str  # rendered lost/delayed-bag guidance, when relevant
    facts: dict  # persisted conversation memory slots (ticket, phone, ...)
    profile: dict  # passenger identity + bookings, for inheritance
    greeting: str  # "greeting" | "thanks" when the turn is small talk
    tool_results: list[dict]  # raw payloads returned by the tools this turn
    answer: str


async def dispatch(state: AgentState) -> dict:
    """Catch small talk before the retrieval pipeline sees it.

    "Hi" has no answer in the knowledge base, and the QA prompt treats a thin
    context as "cannot help" — which used to greet every new conversation with
    a hand-off offer. Greetings are answered deterministically instead.
    """
    reply = greeting_reply(state.get("query", ""))
    if reply is None:
        return {}
    text, options = reply
    return {
        "greeting": "greeting",
        "answer": text,
        "escalation_offer": {"reason": "greeting", "options": options},
    }


def _dispatch_next(state: AgentState) -> str:
    return "greeting" if state.get("greeting") else "chat"


async def greet(state: AgentState) -> dict:
    """Terminal node for a greeting turn: the answer is already set."""
    return {}


async def analyse_sentiment(state: AgentState) -> dict:
    try:
        result = await asyncio.wait_for(
            get_sentiment_analyser().analyse_async(state.get("query", "")),
            timeout=SENTIMENT_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception("Sentiment step failed; assuming neutral")
        result = SentimentResult(label="neutral", score=0.0, confidence=0.0)
    return {"sentiment_score": result.score, "sentiment_label": result.label}


async def classify(state: AgentState) -> dict:
    result = await classify_intent(state.get("query", ""))
    return {"intent": result.intent, "intent_source": result.source}


async def route(state: AgentState) -> dict:
    """Decide escalate vs answer-with-offer vs routine. Reads history from the DB.

    Hard triggers (safety, explicit human request) escalate at once. Soft
    triggers (compensation, a negative-sentiment streak, or a conversation the
    bot cannot close) do NOT replace the answer: the bot answers the passenger
    and attaches a "Connect me to a human" quick reply. The hand-off happens
    only after `soft_escalation_max_offers` unanswered offers.
    """
    intent = state.get("intent", "routine")
    facts = state.get("facts") or {}

    # If we asked for the passenger's email, treat the whole turn as
    # routine (never escalate) even if the LLM misclassified the address or
    # the passenger's phrasing as safety/human_request.
    if facts.get("needs_email"):
        intent = "routine"

    # If we asked for the passenger's phone and they supplied it, treat as
    # routine even if the LLM misclassified the number as safety/human_request.
    if facts.get("pending_booking"):
        query = state.get("query", "") or ""
        if re.search(r"(?:\+?254|0)\s?[17]\d{2}[\s-]?\d{3}[\s-]?\d{3}\b", query):
            intent = "routine"

    hard_intents = set(HARD_ESCALATING_INTENTS)
    if settings.escalate_on_compensation:
        hard_intents |= SOFT_ESCALATING_INTENTS

    if intent in hard_intents:
        reason = f"high-risk intent: {intent}"
        if state.get("intent_source") == "keyword":
            reason = "safety keyword trigger"
        return {"escalate": True, "escalation_reason": reason}

    soft_reason = ""
    if intent in SOFT_ESCALATING_INTENTS:
        soft_reason = f"high-risk intent: {intent}"
    if not soft_reason:
        soft_reason = await _history_triggers(state)

    if not soft_reason:
        return {"escalate": False, "escalation_reason": ""}

    # The passenger already declined a hand-off in this conversation: keep
    # answering, stop pushing.
    if (state.get("facts") or {}).get("human_declined"):
        return {"escalate": False, "escalation_reason": soft_reason}

    # Soft trigger fired. Escalate outright when soft escalation is disabled or
    # the passenger has already been offered options enough times.
    if not settings.soft_escalation_enabled:
        return {"escalate": True, "escalation_reason": soft_reason}
    if state.get("soft_offers", 0) >= settings.soft_escalation_max_offers:
        return {"escalate": True, "escalation_reason": soft_reason}

    await _record_soft_offer(state.get("conversation_id"))
    return {"offer_escalation": True, "escalation_reason": soft_reason}


async def _record_soft_offer(conversation_id: str | None) -> None:
    """Count a hand-off offer made in this conversation."""
    if not conversation_id:
        return
    import uuid

    from sqlalchemy import update

    from app.db import SessionLocal
    from app.models import Conversation

    try:
        async with SessionLocal() as db:
            await db.execute(
                update(Conversation)
                .where(Conversation.id == uuid.UUID(str(conversation_id)))
                .values(soft_offers=Conversation.soft_offers + 1)
            )
            await db.commit()
    except Exception:
        logger.exception("Could not record soft-escalation offer")


async def _history_triggers(state: AgentState) -> str:
    """Sentiment-streak and loop triggers, based on stored conversation state."""
    conversation_id = state.get("conversation_id")
    if not conversation_id:
        return ""

    import uuid

    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import ConversationStatus, Message, MessageSender

    try:
        conv_uuid = uuid.UUID(str(conversation_id))
    except ValueError:
        return ""

    # The conversation row was already loaded by run_turn and passed in
    # state; only the message history needs to be queried here.
    if state.get("conversation_status") == ConversationStatus.RESOLVED.value:
        return ""

    async with SessionLocal() as db:
        # Sentiment streak: current turn + preceding passenger turns.
        prior = (
            await db.execute(
                select(Message.sentiment_score)
                .where(
                    Message.conversation_id == conv_uuid,
                    Message.sender == MessageSender.PASSENGER,
                    Message.sentiment_score.is_not(None),
                )
                .order_by(Message.created_at.desc())
                .limit(
                    max(
                        0,
                        settings.sentiment_escalation_consecutive_turns - 1,
                    )
                )
            )
        ).scalars().all()

        streak_scores = list(prior)
        current = state.get("sentiment_score")
        if current is not None:
            streak_scores.insert(0, current)
        needed = settings.sentiment_escalation_consecutive_turns
        window = streak_scores[:needed]
        if len(window) == needed and all(
            s < settings.sentiment_escalation_threshold for s in window
        ):
            return f"sentiment below {settings.sentiment_escalation_threshold} for {needed} consecutive turns"

        # Loop trigger: still non-routine after max_loop_turns.
        if (
            state.get("conversation_turn_count", 0) > settings.max_loop_turns
            and state.get("intent", "routine") != "routine"
        ):
            return f"unresolved for more than {settings.max_loop_turns} turns"

    return ""


async def retrieve(state: AgentState) -> dict:
    chunks = await get_retriever().retrieve(state.get("query", ""))
    return {"chunks": [chunk.__dict__ for chunk in chunks]}


def _format_context(chunks: list[dict]) -> tuple[str, str]:
    cap = settings.context_max_chars
    parts, titles = [], []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title") or chunk.get("source_url") or f"source {i}"
        titles.append(title)
        content = chunk.get("content", "")
        if cap and len(content) > cap:
            content = content[:cap].rstrip() + "…"
        parts.append(f"[{i}] {title}\n{content}")
    return "\n\n".join(parts), ", ".join(dict.fromkeys(titles)) or "none"


async def flight_lookup(state: AgentState) -> dict:
    """Fetch mock flight status/delays when the query asks about a flight.

    Detection is deterministic (`detect_flight_request`), so the node only
    runs pipeline work on genuine flight questions; everything else returns
    an empty update and the graph continues unchanged. Failures degrade to
    "no flight context" rather than failing the turn.
    """
    request = detect_flight_request(state.get("query", ""))
    if request is None:
        return {}
    lookup = get_flight_lookup()
    try:
        if request["type"] == "status":
            info = await lookup.status(request["flight_no"])
            context = lookup.render_status(info)
        else:
            rows = await lookup.delays(request.get("route"))
            context = lookup.render_delays(rows)
            if not context:
                context = (
                    f"No flights on {request.get('route') or 'those routes'} "
                    "are delayed right now."
                )
    except Exception:
        logger.exception("Flight lookup failed; continuing without flight context")
        return {}
    return {"flight_context": context}


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


async def baggage_lookup(state: AgentState) -> dict:
    """Attach lost/delayed-bag guidance when the query is about a missing bag.

    Deterministic detection and in-process content: no external call, no KB
    lookup needed. The QA node relays the three options and offers a human.
    """
    if not detect_baggage_request(state.get("query", "")):
        return {}
    return {"baggage_context": render_guidance()}


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


async def escalate(state: AgentState) -> dict:
    """Create the Escalation row, flag the conversation, and acknowledge it.

    The acknowledgement matters: the passenger used to be handed to a human
    with no bot message at all, which read as "the bot gave up on my first
    question".
    """
    import uuid

    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import (
        Conversation,
        ConversationStatus,
        Escalation,
        EscalationStatus,
        Message,
    )

    conversation_id = state.get("conversation_id")
    if not conversation_id:
        logger.error("Escalation without conversation_id; dropping")
        return {}

    conv_uuid = uuid.UUID(str(conversation_id))
    async with SessionLocal() as db:
        conversation = (
            await db.execute(select(Conversation).where(Conversation.id == conv_uuid))
        ).scalar_one_or_none()
        if conversation is None:
            logger.error("Escalation target conversation %s not found", conversation_id)
            return {}

        recent = (
            await db.execute(
                select(Message)
                .where(Message.conversation_id == conv_uuid)
                .order_by(Message.created_at.desc())
                .limit(SUMMARY_MESSAGES_FOR_ESCALATION)
            )
        ).scalars().all()
        summary = "\n".join(
            f"{m.sender.value}: {m.content[:300]}" for m in reversed(recent)
        )

        already_open = (
            await db.execute(
                select(Escalation).where(
                    Escalation.conversation_id == conv_uuid,
                    Escalation.status.in_([EscalationStatus.PENDING, EscalationStatus.ASSIGNED]),
                )
            )
        ).scalar_one_or_none()

        created = already_open is None
        if created:
            db.add(
                Escalation(
                    conversation_id=conv_uuid,
                    reason=state.get("escalation_reason", ""),
                    summary=summary[:4000],
                )
            )
        conversation.status = ConversationStatus.ESCALATED
        conversation.escalation_reason = state.get("escalation_reason", "")
        await db.commit()

    return {
        "escalate": True,
        "escalation_created": created,
        "answer": _escalation_ack(state),
    }


def _escalation_ack(state: AgentState) -> str:
    """The bot message sent alongside a hand-off."""
    intent = state.get("intent", "")
    if intent == "safety":
        return ESCALATION_ACK_SAFETY
    if intent == "human_request":
        return ESCALATION_ACK_HUMAN
    return ESCALATION_ACK


def _route_next(state: AgentState) -> str:
    """Route to the human, or on to the answer path.

    A soft trigger no longer diverts the turn: the answer is produced first
    and the hand-off is offered as a quick reply beside it.
    """
    if state.get("escalate"):
        return "escalate"
    return "retrieve"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("dispatch", dispatch)
    graph.add_node("greet", greet)
    graph.add_node("analyse_sentiment", analyse_sentiment)
    graph.add_node("classify", classify)
    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("flight_lookup", flight_lookup)
    graph.add_node("baggage_lookup", baggage_lookup)
    graph.add_node("generate_answer", answer)
    graph.add_node("handle_escalation", escalate)

    graph.set_entry_point("dispatch")
    graph.add_conditional_edges(
        "dispatch",
        _dispatch_next,
        {"greeting": "greet", "chat": "analyse_sentiment"},
    )
    graph.add_edge("greet", END)
    graph.add_edge("analyse_sentiment", "classify")
    graph.add_edge("classify", "route")
    graph.add_conditional_edges(
        "route",
        _route_next,
        {
            "escalate": "handle_escalation",
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("retrieve", "flight_lookup")
    graph.add_edge("flight_lookup", "baggage_lookup")
    graph.add_edge("baggage_lookup", "generate_answer")
    graph.add_edge("generate_answer", END)
    graph.add_edge("handle_escalation", END)
    return graph.compile()


_graph = None


def get_graph():
    """Shared compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
