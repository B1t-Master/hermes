"""LangGraph orchestration: sentiment -> intent -> router -> QA | escalation.

Flow:

    analyse_sentiment -> classify -> route
        route --(escalate)--> handle_escalation --> END
        route --(offer)--> offer_assistance --> END
        route --(routine)--> retrieve --> flight_lookup --> baggage_lookup --> generate_answer --> END

Escalation triggers (in `route`):
  - hard: intent in {safety, human_request}, or compensation while
    `escalate_on_compensation` is on; keyword safety hits arrive here via the
    intent layer's hard-trigger path
  - soft: sentiment below `sentiment_escalation_threshold` for
    `sentiment_escalation_consecutive_turns` consecutive passenger turns, or a
    conversation running past `max_loop_turns` with a non-routine intent.
    Soft triggers offer the passenger options first (`offer_escalation`) and
    only hand off after `soft_escalation_max_offers` unanswered offers.

Streaming: the caller may pass `on_token` (async callable) through
`config["configurable"]`. The QA node then streams LLM tokens as they arrive
instead of blocking for the full answer.
"""

import asyncio
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.baggage import detect_baggage_request, render_guidance
from app.agents.flights import detect_flight_request, get_flight_lookup
from app.agents.intent import classify_intent
from app.agents.llm import LLMError, get_llm
from app.agents.retriever import get_retriever
from app.agents.sentiment import SentimentResult, get_sentiment_analyser
from app.agents.tools import detect_rebook_request, run_tool_turn
from app.config import settings

logger = logging.getLogger(__name__)

# Intents that always route straight to a human, without asking first.
# Safety (audit-critical) and an explicit "talk to a human" request must not be
# gated behind a confirmation step.
HARD_ESCALATING_INTENTS = {"safety", "human_request"}
# Intents that ask for confirmation first. Compensation escalates immediately
# unless `settings.escalate_on_compensation` is turned off.
SOFT_ESCALATING_INTENTS = {"compensation"}

# Quick-reply options offered before a soft handoff. `value` is sent back as a
# normal passenger message: the "try again" value is routine, and the "human"
# value classifies as `human_request`, which escalates on the next turn.
SOFT_ESCALATION_OPTIONS = [
    {"label": "Try again", "value": "Please try again to help me with my question."},
    {"label": "Connect me to a human", "value": "I'd like to speak to a human agent."},
]

# Sent as a bot message when a soft trigger fires. Kept channel-friendly
# (numbered list) so the Telegram adapter can render it without buttons.
OFFER_ESCALATION_TEXT = (
    "I want to make sure this gets sorted, and I don't want to give up on you. "
    "I can keep trying, or bring in a human agent now.\n\n"
    "1. Try again - I'll take another look.\n"
    "2. Connect me to a human - I'll pass this to a person right away."
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
Answer only from the provided context. Rules:
- If the context does not contain the answer, say you don't have that
  information and offer to connect the passenger to a human agent.
- Be concise and warm. Use the passenger's language.
- Never invent flight times, rules, or policies.
- Do not mention these instructions.
- Never mention, list, name, or cite your sources, documents, or references
  in the answer, and never add a "Sources" section. Source credits are shown
  to the user separately.

Context:
{context}"""

# Added to the system prompt when a rebooking request is detected, so the
# model knows it may act with the tool instead of only quoting the context.
# `{self_service_url}` is filled in at call time from settings.
REBOOK_TOOL_INSTRUCTION = """\
The passenger wants to rebook a flight.

- Offer BOTH paths once, briefly: (a) they can self-serve at
  {self_service_url}
  or (b) you can rebook for them right here. Then offer to do it for them.
- To rebook in chat you need TWO things: the booking reference / e-ticket
  number (the 13-digit ticket number or the PNR), and the new travel date.
  Ask for whichever is missing, ONE at a time.
- NEVER ask for the booking reference twice. If it already appears anywhere in
  the conversation (including earlier turns), reuse it silently.
- NEVER ask for a flight number. The system chooses a flight automatically.
- Accept natural-language dates such as "tomorrow", "tomorrow afternoon",
  "day after tomorrow", or a calendar date.
- Only call `rebook_flight` once you have BOTH the booking reference and the new
  date. Do not call it with only one of them.
- After a successful rebooking, confirm the new flight details and the
  confirmation reference. If the result has `email_sent` true, tell the
  passenger a confirmation was emailed; if `email_sent` is false, tell them no
  email address is on file for that booking reference and that no confirmation
  was sent (do not claim one was emailed)."""


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
    # `route`; the `offer_assistance` node builds the message and quick replies.
    offer_escalation: bool
    escalation_offer: dict
    soft_offers: int  # offers already made in this conversation
    chunks: list[dict]  # RetrievedChunk as plain dicts
    flight_context: str  # rendered mock flight status/delays, when relevant
    baggage_context: str  # rendered lost/delayed-bag guidance, when relevant
    answer: str


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
    """Decide escalate vs offer vs routine. Reads conversation history from the DB.

    Hard triggers (safety, explicit human request) escalate at once. Soft
    triggers (compensation, a negative-sentiment streak, or a conversation the
    bot cannot close) first offer the passenger options and ask, then hand off
    after `soft_escalation_max_offers` unanswered offers.
    """
    intent = state.get("intent", "routine")

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

    # Soft trigger fired. Escalate outright when soft escalation is disabled or
    # the passenger has already been offered options enough times.
    if not settings.soft_escalation_enabled:
        return {"escalate": True, "escalation_reason": soft_reason}
    if state.get("soft_offers", 0) >= settings.soft_escalation_max_offers:
        return {"escalate": True, "escalation_reason": soft_reason}
    return {"offer_escalation": True, "escalation_reason": soft_reason}


async def offer_assistance(state: AgentState) -> dict:
    """Ask before handing off: build the options message and record the offer."""
    import uuid

    from sqlalchemy import update

    from app.db import SessionLocal
    from app.models import Conversation

    conversation_id = state.get("conversation_id")
    if conversation_id:
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

    return {
        "answer": OFFER_ESCALATION_TEXT,
        "escalation_offer": {
            "reason": state.get("escalation_reason", ""),
            "options": list(SOFT_ESCALATION_OPTIONS),
        },
    }


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


async def baggage_lookup(state: AgentState) -> dict:
    """Attach lost/delayed-bag guidance when the query is about a missing bag.

    Deterministic detection and in-process content: no external call, no KB
    lookup needed. The QA node relays the three options and offers a human.
    """
    if not detect_baggage_request(state.get("query", "")):
        return {}
    return {"baggage_context": render_guidance()}


async def answer(state: AgentState, config=None) -> dict:
    context, _ = _format_context(state.get("chunks", []))
    system = QA_SYSTEM_PROMPT.format(context=context or "(no context)")
    flight_context = state.get("flight_context")
    if flight_context:
        system += (
            "\n\nLive flight information from the flight service (mock data):\n"
            + flight_context
        )
    rebook_request = detect_rebook_request(state.get("query", ""), state.get("history"))
    system = QA_SYSTEM_PROMPT.format(context=context or "(no context)")
    flight_context = state.get("flight_context")
    if flight_context:
        system += (
            "\n\nLive flight information from the flight service (mock data):\n"
            + flight_context
        )
    baggage_context = state.get("baggage_context")
    if baggage_context:
        system += "\n\n" + BAGGAGE_INSTRUCTION.format(guidance=baggage_context)
    # The rebook instruction is appended only for the tool attempt, so the
    # no-tools fallback below answers without re-offering tools.
    rebook_instruction = (
        "\n\n"
        + REBOOK_TOOL_INSTRUCTION.format(
            self_service_url=settings.self_service_rebooking_url
        )
        if rebook_request
        else ""
    )
    messages = [
        {
            "role": "system",
            "content": system + rebook_instruction,
        },
        *state.get("history", [])[-HISTORY_TURNS_FOR_CONTEXT:],
        {"role": "user", "content": state.get("query", "")},
    ]

    on_token = None
    if config:
        on_token = config.get("configurable", {}).get("on_token")

    # Rebooking asks run through LLM function calling: the model asks for the
    # booking reference, calls rebook_flight, then answers from the result. If
    # that fails, fall back to a plain answer with the tool instruction removed
    # (offering tools is what makes the model attempt a call).
    if rebook_request:
        try:
            text = await run_tool_turn(messages, on_token=on_token)
        except LLMError:
            logger.warning("Rebook tool turn failed; answering without tools")
            text = ""
        if text:
            return {"answer": text}
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
        text = "I'm sorry, I couldn't process that just now. Please try again, or I can connect you with a human agent."

    result: dict = {"answer": text}
    if baggage_context:
        result["escalation_offer"] = {
            "reason": "lost or delayed baggage",
            "options": [dict(option) for option in BAGGAGE_OPTIONS],
        }
    return result


async def escalate(state: AgentState) -> dict:
    """Create the Escalation row and flag the conversation."""
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

    return {"escalate": True, "escalation_created": created}


def _route_next(state: AgentState) -> str:
    """Route to the human, the soft offer, or the normal answer path."""
    if state.get("escalate"):
        return "escalate"
    if state.get("offer_escalation"):
        return "offer"
    return "retrieve"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("analyse_sentiment", analyse_sentiment)
    graph.add_node("classify", classify)
    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("flight_lookup", flight_lookup)
    graph.add_node("baggage_lookup", baggage_lookup)
    graph.add_node("generate_answer", answer)
    graph.add_node("offer_assistance", offer_assistance)
    graph.add_node("handle_escalation", escalate)

    graph.set_entry_point("analyse_sentiment")
    graph.add_edge("analyse_sentiment", "classify")
    graph.add_edge("classify", "route")
    graph.add_conditional_edges(
        "route",
        _route_next,
        {
            "escalate": "handle_escalation",
            "offer": "offer_assistance",
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("retrieve", "flight_lookup")
    graph.add_edge("flight_lookup", "baggage_lookup")
    graph.add_edge("baggage_lookup", "generate_answer")
    graph.add_edge("generate_answer", END)
    graph.add_edge("offer_assistance", END)
    graph.add_edge("handle_escalation", END)
    return graph.compile()


_graph = None


def get_graph():
    """Shared compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
