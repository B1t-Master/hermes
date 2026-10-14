"""Room + dashboard fan-out for a finished turn.

The WebSocket handler and the Telegram adapter both run the same
``run_turn`` pipeline; this module owns the part where the result is
broadcast so an escalation raised on any channel reaches the agent
dashboard identically (and a watching agent sees the passenger's turn).

The caller keeps whatever is channel-specific: streaming tokens to a
socket, or sending the reply back over Telegram.
"""

import logging
import uuid

from sqlalchemy import select

from app.agents.hub import conversation_hub, dashboard_hub
from app.db import SessionLocal
from app.models import Escalation

logger = logging.getLogger(__name__)


async def _latest_escalation_id(conversation_id: uuid.UUID) -> str | None:
    async with SessionLocal() as db:
        escalation = (
            await db.execute(
                select(Escalation)
                .where(Escalation.conversation_id == conversation_id)
                .order_by(Escalation.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    return str(escalation.id) if escalation else None


async def notify_turn(conversation_id: uuid.UUID, result: dict) -> None:
    """Broadcast a finished turn to the conversation room and dashboards.

    Mirrors the ordering in the WebSocket handler: the passenger message
    first, then either a new escalation (room event + queue update) or the
    bot answer. Best-effort: a fan-out failure must not fail the turn.
    """
    try:
        await conversation_hub.send(
            conversation_id,
            {"type": "passenger_message", "message": result["passenger_message"]},
        )

        if result.get("escalate"):
            # Only announce a genuinely new escalation. A reply in an already
            # escalated conversation routes to the agent without re-escalating.
            if result.get("escalation_created"):
                await conversation_hub.send(
                    conversation_id,
                    {"type": "escalated", "reason": result.get("escalation_reason") or ""},
                )
                escalation_id = await _latest_escalation_id(conversation_id)
                await dashboard_hub.broadcast(
                    {
                        "type": "queue_update",
                        "action": "created",
                        "escalation_id": escalation_id,
                        "conversation_id": str(conversation_id),
                    }
                )
            return

        if result.get("bot_message"):
            await conversation_hub.send(
                conversation_id, {"type": "bot_message", "message": result["bot_message"]}
            )
    except Exception:
        logger.exception("Turn fan-out failed for conversation=%s", conversation_id)
