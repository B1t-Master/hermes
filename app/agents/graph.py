"""LangGraph orchestration: sentiment -> intent -> router -> QA | escalation.

Flow:

    analyse_sentiment -> classify -> route
        route --(escalate)--> handle_escalation --> END
        route --(routine)--> retrieve --> generate_answer --> END

Escalation triggers (in `route`):
  - intent in {safety, compensation, human_request} (keyword safety hits
    arrive here via the intent layer's hard-trigger path)
  - sentiment below `sentiment_escalation_threshold` for
    `sentiment_escalation_consecutive_turns` consecutive passenger turns
  - conversation running past `max_loop_turns` with a non-routine intent
    (the passenger keeps bringing something the bot can't close)

Streaming: the caller may pass `on_token` (async callable) through
`config["configurable"]`. The QA node then streams LLM tokens as they arrive
instead of blocking for the full answer.
"""

import asyncio
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.intent import classify_intent
from app.agents.llm import LLMError, get_llm
from app.agents.retriever import get_retriever
from app.agents.sentiment import SentimentResult, get_sentiment_analyser
from app.config import settings

logger = logging.getLogger(__name__)

# Intents that route straight to a human.
ESCALATING_INTENTS = {"safety", "compensation", "human_request"}

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

Context:
{context}

Relevant sources to cite inline as [title] when you use them: {sources}"""


class AgentState(TypedDict, total=False):
    conversation_id: str
    passenger_id: str
    query: str
    history: list[dict]  # [{role, content}, ...] most recent last
    sentiment_score: float
    sentiment_label: str
    intent: str
    intent_source: str
    escalate: bool
    escalation_reason: str
    chunks: list[dict]  # RetrievedChunk as plain dicts
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
    """Decide escalate vs routine. Reads conversation history from the DB."""
    intent = state.get("intent", "routine")

    if intent in ESCALATING_INTENTS:
        reason = f"high-risk intent: {intent}"
        if state.get("intent_source") == "keyword":
            reason = "safety keyword trigger"
        return {"escalate": True, "escalation_reason": reason}

    triggers = await _history_triggers(state)
    if triggers:
        return {"escalate": True, "escalation_reason": triggers}
    return {"escalate": False, "escalation_reason": ""}


async def _history_triggers(state: AgentState) -> str:
    """Sentiment-streak and loop triggers, based on stored conversation state."""
    conversation_id = state.get("conversation_id")
    if not conversation_id:
        return ""

    import uuid

    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import Conversation, ConversationStatus, Message, MessageSender

    try:
        conv_uuid = uuid.UUID(str(conversation_id))
    except ValueError:
        return ""

    async with SessionLocal() as db:
        conversation = (
            await db.execute(select(Conversation).where(Conversation.id == conv_uuid))
        ).scalar_one_or_none()
        if conversation is None or conversation.status == ConversationStatus.RESOLVED:
            return ""

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
            conversation.turn_count > settings.max_loop_turns
            and state.get("intent", "routine") != "routine"
        ):
            return f"unresolved for more than {settings.max_loop_turns} turns"

    return ""


async def retrieve(state: AgentState) -> dict:
    chunks = await get_retriever().retrieve(state.get("query", ""))
    return {"chunks": [chunk.__dict__ for chunk in chunks]}


def _format_context(chunks: list[dict]) -> tuple[str, str]:
    parts, titles = [], []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title") or chunk.get("source_url") or f"source {i}"
        titles.append(title)
        parts.append(f"[{i}] {title}\n{chunk.get('content', '')}")
    return "\n\n".join(parts), ", ".join(dict.fromkeys(titles)) or "none"


async def answer(state: AgentState, config=None) -> dict:
    context, sources = _format_context(state.get("chunks", []))
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT.format(context=context or "(no context)", sources=sources)},
        *state.get("history", [])[-HISTORY_TURNS_FOR_CONTEXT:],
        {"role": "user", "content": state.get("query", "")},
    ]

    on_token = None
    if config:
        on_token = config.get("configurable", {}).get("on_token")

    llm = get_llm()
    try:
        if on_token is not None:
            parts: list[str] = []
            async for token in llm.stream(messages, max_tokens=1024):
                parts.append(token)
                await on_token(token)
            text = "".join(parts)
        else:
            text = await llm.complete(messages, max_tokens=1024)
    except LLMError as exc:
        logger.error("QA generation failed: %s", exc)
        text = "I'm sorry, I couldn't process that just now. Please try again, or I can connect you with a human agent."

    if state.get("chunks"):
        titles = _format_context(state["chunks"])[1]
        text += f"\n\nSources: {titles}"

    return {"answer": text}


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

        if already_open is None:
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

    return {"escalate": True}


def _should_escalate(state: AgentState) -> str:
    return "escalate" if state.get("escalate") else "retrieve"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("analyse_sentiment", analyse_sentiment)
    graph.add_node("classify", classify)
    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", answer)
    graph.add_node("handle_escalation", escalate)

    graph.set_entry_point("analyse_sentiment")
    graph.add_edge("analyse_sentiment", "classify")
    graph.add_edge("classify", "route")
    graph.add_conditional_edges(
        "route",
        _should_escalate,
        {"escalate": "handle_escalation", "retrieve": "retrieve"},
    )
    graph.add_edge("retrieve", "generate_answer")
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
