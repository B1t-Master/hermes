"""Agent dashboard API: escalation queue + claim/release/resolve workflow.

Claim is atomic (`UPDATE ... WHERE status = 'pending' RETURNING`), so two
agents clicking at the same time produce exactly one winner: the loser gets
409 and refetches the queue. Each mutation broadcasts a `queue_update` event
to every connected `/ws/dashboard` socket.

Transcript view (`GET .../messages`) is open to any authenticated agent so a
pending card can be inspected before claiming; *replying* requires the claim
(checked on every message by the conversation WS).
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.hub import dashboard_hub
from app.db import get_db
from app.deps import get_current_agent
from app.models import (
    Agent,
    Conversation,
    ConversationStatus,
    Escalation,
    EscalationStatus,
    Message,
    Passenger,
)
from app.schemas import EscalationOut, MessageOut, QueueEscalationOut

router = APIRouter(prefix="/agents", tags=["agents"])

_STATUS_VALUES = ("active", "pending", "assigned", "resolved", "all")


async def _broadcast(escalation: Escalation, action: str) -> None:
    await dashboard_hub.broadcast(
        {
            "type": "queue_update",
            "action": action,
            "escalation_id": str(escalation.id),
            "conversation_id": str(escalation.conversation_id),
        }
    )


@router.get("/escalations", response_model=list[QueueEscalationOut])
async def list_escalations(
    status: str = Query("active", pattern="^(active|pending|assigned|resolved|all)$"),
    agent: Agent = Depends(get_current_agent),
    db: AsyncSession = Depends(get_db),
):
    """Queue listing. `active` (default) = pending + assigned, oldest first."""

    latest_message = (
        select(Message.content)
        .where(Message.conversation_id == Conversation.id)
        .order_by(Message.created_at.desc())
        .limit(1)
    ).correlate(Conversation)

    stmt = (
        select(
            Escalation,
            Conversation,
            Passenger.first_name,
            latest_message.label("last_message"),
        )
        .join(Conversation, Escalation.conversation_id == Conversation.id)
        .join(Passenger, Conversation.passenger_id == Passenger.id)
        .order_by(Escalation.created_at.asc())
    )
    if status == "pending":
        stmt = stmt.where(Escalation.status == EscalationStatus.PENDING)
    elif status == "assigned":
        stmt = stmt.where(Escalation.status == EscalationStatus.ASSIGNED)
    elif status == "resolved":
        stmt = stmt.where(Escalation.status == EscalationStatus.RESOLVED)
    elif status == "active":
        stmt = stmt.where(
            Escalation.status.in_([EscalationStatus.PENDING, EscalationStatus.ASSIGNED])
        )
    # "all" -> no filter

    rows = (await db.execute(stmt)).all()

    # Agent names for assigned cards (single extra query, not N+1).
    agent_ids = {esc.agent_id for esc, *_ in rows if esc.agent_id}
    agent_names: dict[uuid.UUID, str] = {}
    if agent_ids:
        for row in (
            await db.execute(select(Agent.id, Agent.name).where(Agent.id.in_(agent_ids)))
        ).all():
            agent_names[row[0]] = row[1]

    return [
        QueueEscalationOut(
            id=esc.id,
            conversation_id=esc.conversation_id,
            reason=esc.reason,
            summary=esc.summary,
            status=esc.status.value,
            created_at=esc.created_at,
            passenger_name=first_name,
            turn_count=conv.turn_count,
            conversation_status=conv.status.value,
            agent_id=esc.agent_id,
            agent_name=agent_names.get(esc.agent_id) if esc.agent_id else None,
            last_message=last_message,
        )
        for esc, conv, first_name, last_message in rows
    ]


@router.get("/escalations/{escalation_id}/messages", response_model=list[MessageOut])
async def escalation_messages(
    escalation_id: uuid.UUID,
    agent: Agent = Depends(get_current_agent),
    db: AsyncSession = Depends(get_db),
):
    escalation = (
        await db.execute(select(Escalation).where(Escalation.id == escalation_id))
    ).scalar_one_or_none()
    if escalation is None:
        raise HTTPException(status_code=404, detail="Escalation not found")
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == escalation.conversation_id)
        .order_by(Message.created_at.asc())
    )
    return result.scalars().all()


@router.post("/escalations/{escalation_id}/claim", response_model=EscalationOut)
async def claim_escalation(
    escalation_id: uuid.UUID,
    agent: Agent = Depends(get_current_agent),
    db: AsyncSession = Depends(get_db),
):
    """Atomically take a pending escalation. Exactly one claimant can win."""
    stmt = (
        update(Escalation)
        .where(
            Escalation.id == escalation_id,
            Escalation.status == EscalationStatus.PENDING,
        )
        .values(status=EscalationStatus.ASSIGNED, agent_id=agent.id)
        .returning(Escalation.id)
    )
    claimed_id = (await db.execute(stmt)).scalar_one_or_none()
    if claimed_id is None:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Escalation already claimed or closed")
    await db.commit()
    escalation = (
        await db.execute(select(Escalation).where(Escalation.id == claimed_id))
    ).scalar_one()
    await _broadcast(escalation, "claimed")
    return escalation


@router.post("/escalations/{escalation_id}/release", response_model=EscalationOut)
async def release_escalation(
    escalation_id: uuid.UUID,
    agent: Agent = Depends(get_current_agent),
    db: AsyncSession = Depends(get_db),
):
    """Hand a claimed escalation back to the queue."""
    stmt = (
        update(Escalation)
        .where(
            Escalation.id == escalation_id,
            Escalation.status == EscalationStatus.ASSIGNED,
            Escalation.agent_id == agent.id,
        )
        .values(status=EscalationStatus.PENDING, agent_id=None)
        .returning(Escalation.id)
    )
    released_id = (await db.execute(stmt)).scalar_one_or_none()
    if released_id is None:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Not your assigned escalation")
    await db.commit()
    escalation = (
        await db.execute(select(Escalation).where(Escalation.id == released_id))
    ).scalar_one()
    await _broadcast(escalation, "released")
    return escalation


@router.post("/escalations/{escalation_id}/resolve", response_model=EscalationOut)
async def resolve_escalation(
    escalation_id: uuid.UUID,
    agent: Agent = Depends(get_current_agent),
    db: AsyncSession = Depends(get_db),
):
    """Close the escalation and the conversation (passenger must start a new one)."""
    escalation = (
        await db.execute(
            select(Escalation).where(Escalation.id == escalation_id)
        )
    ).scalar_one_or_none()
    if escalation is None:
        raise HTTPException(status_code=404, detail="Escalation not found")
    if escalation.status != EscalationStatus.ASSIGNED or escalation.agent_id != agent.id:
        raise HTTPException(status_code=409, detail="Claim the escalation before resolving")

    escalation.status = EscalationStatus.RESOLVED
    conversation = (
        await db.execute(
            select(Conversation).where(Conversation.id == escalation.conversation_id)
        )
    ).scalar_one_or_none()
    if conversation is not None:
        conversation.status = ConversationStatus.RESOLVED
    await db.commit()
    await db.refresh(escalation)

    # Tell the passenger's room their conversation was closed.
    from app.agents.hub import conversation_hub

    await conversation_hub.send(
        escalation.conversation_id,
        {"type": "resolved", "escalation_id": str(escalation.id)},
    )
    await _broadcast(escalation, "resolved")
    return escalation
