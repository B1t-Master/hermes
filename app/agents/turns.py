"""One passenger turn = persist input -> run graph -> persist outputs.

Shared by the WebSocket handler (streaming) and the REST fallback
(non-streaming), so there is exactly one pipeline to reason about.

Ordering note: the passenger message is persisted *after* the graph run,
because the graph's sentiment-streak trigger reads prior passenger messages
from the DB; persisting first would make the current turn count as its own
history. The tradeoff: a crash mid-graph loses that one inbound message.
"""

import logging
import uuid

from sqlalchemy import select

from app.agents.graph import get_graph
from app.db import SessionLocal
from app.models import Conversation, ConversationStatus, Message, MessageSender

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 10

_ROLE = {MessageSender.PASSENGER.value: "user", MessageSender.BOT.value: "assistant", MessageSender.AGENT.value: "assistant"}


def message_to_dict(message: Message) -> dict:
    return {
        "id": str(message.id),
        "conversation_id": str(message.conversation_id),
        "sender": message.sender.value,
        "content": message.content,
        "sentiment_score": message.sentiment_score,
        "sentiment_label": message.sentiment_label,
        "sources": message.sources or [],
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def citations_from_chunks(chunks: list) -> list[dict]:
    """Deduplicated [{title, url}] pairs from the graph's retrieved chunks.

    The graph stores chunks as plain dicts (RetrievedChunk.__dict__), so keys
    are `title`, `source_url`, `content`, `score`, `topic`, `chunk_index`.
    """
    citations: list[dict] = []
    seen: set[tuple[str, str | None]] = set()
    for chunk in chunks:
        title = chunk.get("title") or chunk.get("source_url") or "Source"
        url = chunk.get("source_url")
        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        citation: dict = {"title": title, "url": url}
        score = chunk.get("score")
        if score is not None:
            citation["score"] = round(score, 4)
        citations.append(citation)
    return citations


async def load_history(conversation_id: uuid.UUID, limit: int = HISTORY_LIMIT) -> list[dict]:
    """Prior turns as chat messages, oldest first."""
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
        ).scalars().all()
    rows = list(reversed(rows))
    return [{"role": _ROLE.get(m.sender.value, "assistant"), "content": m.content} for m in rows]


async def get_conversation(conversation_id: uuid.UUID) -> Conversation | None:
    async with SessionLocal() as db:
        return (
            await db.execute(select(Conversation).where(Conversation.id == conversation_id))
        ).scalar_one_or_none()


async def run_turn(
    conversation_id: uuid.UUID,
    query: str,
    on_token=None,
) -> dict:
    """Run one graph turn. Returns the final graph state plus persisted messages.

    Expected shape of the return value:
      {**state, "passenger_message": {...}, "bot_message": {...} | None}
    """
    conversation = await get_conversation(conversation_id)
    if conversation is None:
        raise LookupError("conversation not found")
    if conversation.status == ConversationStatus.RESOLVED:
        raise PermissionError("conversation is closed")

    # Escalated conversations belong to the human agent: no bot turns.
    if conversation.status == ConversationStatus.ESCALATED:
        async with SessionLocal() as db:
            message = Message(
                conversation_id=conversation_id,
                sender=MessageSender.PASSENGER,
                content=query,
            )
            db.add(message)
            conversation.turn_count += 1
            await db.commit()
            await db.refresh(message)
        return {
            "escalate": True,
            "escalation_reason": conversation.escalation_reason or "",
            "passenger_message": message_to_dict(message),
            "bot_message": None,
        }

    history = await load_history(conversation_id)

    state = {
        "conversation_id": str(conversation_id),
        "passenger_id": str(conversation.passenger_id),
        "query": query,
        "history": history,
    }
    config = {"configurable": {"on_token": on_token}} if on_token is not None else None
    result = await get_graph().ainvoke(state, config=config)
    result = dict(result)

    escalated = bool(result.get("escalate"))
    answer = result.get("answer", "")

    async with SessionLocal() as db:
        passenger_message = Message(
            conversation_id=conversation_id,
            sender=MessageSender.PASSENGER,
            content=query,
            sentiment_score=result.get("sentiment_score"),
            sentiment_label=result.get("sentiment_label"),
        )
        db.add(passenger_message)
        bot_message = None
        if not escalated and answer:
            bot_message = Message(
                conversation_id=conversation_id,
                sender=MessageSender.BOT,
                content=answer,
                sources=citations_from_chunks(result.get("chunks") or []),
            )
            db.add(bot_message)
        conversation = (
            await db.execute(select(Conversation).where(Conversation.id == conversation_id))
        ).scalar_one()
        conversation.turn_count += 1
        await db.commit()
        await db.refresh(passenger_message)
        if bot_message is not None:
            await db.refresh(bot_message)

    result["passenger_message"] = message_to_dict(passenger_message)
    result["bot_message"] = message_to_dict(bot_message) if bot_message else None
    return result
