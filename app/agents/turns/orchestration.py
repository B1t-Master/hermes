"""One passenger turn = persist input -> run graph -> persist outputs.

Shared by the WebSocket handler (streaming) and the REST fallback
(non-streaming), so there is exactly one pipeline to reason about.

Ordering note: the passenger message is persisted *after* the graph run,
because the graph's sentiment-streak trigger reads prior passenger messages
from the DB; persisting first would make the current turn count as its own
history. The tradeoff: a crash mid-graph loses that one inbound message.
"""

from __future__ import annotations

import time
import uuid

from sqlalchemy import select, update

from app.agents.facts import (
    apply_booking_result,
    apply_rebook_result,
    detect_human_decline,
    extract_facts,
)
from app.agents.graph import get_graph
from app.db import SessionLocal
from app.metrics import ESCALATION, RESPONSE_TIME_MS, RETRIEVAL_HITS, add_metric
from app.models import (
    Conversation,
    ConversationStatus,
    Message,
    MessageSender,
)

from .store import (
    get_conversation,
    load_history,
    load_profile,
    message_to_dict,
)
from .text import citations_from_chunks, strip_source_section, strip_source_urls


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
            "escalation_created": False,
            "escalation_reason": conversation.escalation_reason or "",
            "passenger_message": message_to_dict(message),
            "bot_message": None,
        }

    history = await load_history(conversation_id)
    # Merge this turn's slots into the conversation memory before the graph
    # runs, so the answer prompt can reuse them (and never re-ask).
    facts = extract_facts(query, conversation.facts or {})
    # Also extract from recent assistant messages to capture rebook results.
    # Assistant messages may only fill slots the passenger has not given: a
    # rebooking confirmation quotes a *confirmation* reference, and letting
    # that overwrite the passenger's ticket number broke "what is my ticket
    # number?" a few turns after a rebooking.
    for m in history[-6:]:
        if m.get("role") == "assistant":
            facts = extract_facts(m.get("content", ""), facts, source="assistant")
    # The passenger declining a hand-off ("no thanks, I'll continue") stops the
    # bot from pushing one on every later turn of the same conversation.
    if detect_human_decline(query):
        facts["human_declined"] = True
    profile = await load_profile(conversation.passenger_id)

    state = {
        "conversation_id": str(conversation_id),
        "passenger_id": str(conversation.passenger_id),
        "query": query,
        "history": history,
        # Pre-loaded conversation facts, so the graph's history triggers do
        # not need to re-fetch the row (one fewer DB round trip per turn).
        "conversation_status": conversation.status.value,
        "conversation_turn_count": conversation.turn_count,
        "soft_offers": conversation.soft_offers,
        "facts": facts,
        "profile": profile,
    }
    config = {"configurable": {"on_token": on_token}} if on_token is not None else None
    started = time.perf_counter()
    result = await get_graph().ainvoke(state, config=config)
    response_time_ms = (time.perf_counter() - started) * 1000.0
    result = dict(result)

    escalation_created = bool(result.get("escalation_created"))
    chunks = result.get("chunks") or []
    answer = strip_source_section(strip_source_urls(result.get("answer", "")))
    result["answer"] = answer

    # A successful rebooking is authoritative: the tool knows the ticket, the
    # confirmation reference, the flight and the date exactly. Write them into
    # the memory so the next question about the booking is answered from the
    # booking, not from whatever the model happened to repeat in prose.
    for tool_result in result.get("tool_results") or []:
        facts = apply_rebook_result(facts, tool_result)
        facts = apply_booking_result(facts, tool_result)

    # A booking tool result carries the payment card to render under the bot
    # message. The card collects the M-Pesa number directly.
    payment_uuid = None
    for tool_result in result.get("tool_results") or []:
        candidate = tool_result.get("payment_id")
        if not candidate:
            continue
        try:
            payment_uuid = uuid.UUID(str(candidate))
        except (ValueError, TypeError):
            continue

    async with SessionLocal() as db:
        # Only the first message carrying a payment id renders a card. If an
        # earlier message already holds this payment, drop the id from this
        # turn so a status/ticket reply cannot add a duplicate card.
        if payment_uuid is not None:
            existing = (
                await db.execute(
                    select(Message.id)
                    .where(
                        Message.conversation_id == conversation_id,
                        Message.payment_id == payment_uuid,
                    )
                    .limit(1)
                )
            ).first()
            if existing is not None:
                payment_uuid = None

        # A card needs a bot message to live under. If the tool produced a
        # payment but the model returned no text, synthesize a short cue.
        if payment_uuid is not None and not answer:
            answer = (
                "Your booking is held. Enter your M-Pesa number in the payment "
                "card below to receive the prompt."
            )

        passenger_message = Message(
            conversation_id=conversation_id,
            sender=MessageSender.PASSENGER,
            content=query,
            sentiment_score=result.get("sentiment_score"),
            sentiment_label=result.get("sentiment_label"),
        )
        db.add(passenger_message)
        bot_message = None
        # The bot always answers now, including on the escalation path (the
        # acknowledgement). A hand-off used to leave the passenger with no
        # reply at all.
        if answer:
            escalation_offer = result.get("escalation_offer") or {}
            bot_message = Message(
                conversation_id=conversation_id,
                sender=MessageSender.BOT,
                content=answer,
                sources=citations_from_chunks(result.get("chunks") or []),
                actions=escalation_offer.get("options") or None,
                payment_id=payment_uuid,
            )
            db.add(bot_message)
        add_metric(db, conversation_id, RESPONSE_TIME_MS, response_time_ms)
        add_metric(db, conversation_id, RETRIEVAL_HITS, len(chunks))
        if escalation_created:
            add_metric(db, conversation_id, ESCALATION, 1.0)
        # Increment the counter in-place; avoids a select-then-update.
        await db.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(turn_count=Conversation.turn_count + 1, facts=facts)
        )
        await db.commit()
        await db.refresh(passenger_message)
        if bot_message is not None:
            await db.refresh(bot_message)

    result["passenger_message"] = message_to_dict(passenger_message)
    result["bot_message"] = message_to_dict(bot_message) if bot_message else None
    return result
