"""Escalation node: create the queue row and acknowledge the passenger."""

import logging

from ..constants import (
    ESCALATION_ACK,
    ESCALATION_ACK_HUMAN,
    ESCALATION_ACK_SAFETY,
    SUMMARY_MESSAGES_FOR_ESCALATION,
)
from ..state import AgentState

logger = logging.getLogger(__name__)


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
