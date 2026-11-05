"""Routing node: decide escalate vs answer-with-offer vs routine."""

import logging
import re

from app.config import settings

from ..constants import HARD_ESCALATING_INTENTS, SOFT_ESCALATING_INTENTS
from ..state import AgentState

logger = logging.getLogger(__name__)


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
        # Resolve through the package namespace so the historical
        # ``app.agents.graph._history_triggers`` patch point keeps working.
        from app.agents import graph as graph_pkg

        soft_reason = await graph_pkg._history_triggers(state)

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

    # While we are waiting for the passenger's email address, never
    # escalate — the next turn is the email capture, not a new issue.
    if (state.get("facts") or {}).get("needs_email"):
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
