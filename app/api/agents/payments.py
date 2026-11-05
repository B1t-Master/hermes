"""Dashboard payment state and e-ticket delivery."""

from __future__ import annotations

import uuid

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import payments
from app.agents.hub import conversation_hub
from app.agents.turns import message_to_dict
from app.db import get_db
from app.deps import get_current_agent
from app.models import (
    Agent,
    Escalation,
    EscalationStatus,
    Message,
    MessageSender,
    Payment,
)
from app.schemas import MessageOut, PaymentOut, SendTicketOut
from services import booking

from .common import router


@router.get("/payments/{payment_id}", response_model=PaymentOut)
async def get_payment(
    payment_id: uuid.UUID,
    _: Agent = Depends(get_current_agent),
    db: AsyncSession = Depends(get_db),
):
    """Read-only payment state for the dashboard transcript."""
    payment = (
        await db.execute(select(Payment).where(Payment.id == payment_id))
    ).scalar_one_or_none()
    if payment is None:
        raise HTTPException(status_code=404, detail="Payment not found")
    payload = await payments.get_payment_payload(payment.id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Payment not found")
    return PaymentOut(**payload)


@router.post("/payments/{payment_id}/send-ticket", response_model=SendTicketOut)
async def send_ticket(
    payment_id: uuid.UUID,
    agent: Agent = Depends(get_current_agent),
    db: AsyncSession = Depends(get_db),
):
    """Issue the e-ticket and deliver it in the conversation.

    Only the agent who currently owns the conversation's escalation may do
    this: the ticket is sent as that agent, not as the bot.
    """
    payment = (
        await db.execute(select(Payment).where(Payment.id == payment_id))
    ).scalar_one_or_none()
    if payment is None or payment.conversation_id is None:
        raise HTTPException(status_code=404, detail="Payment not found")

    escalation = (
        await db.execute(
            select(Escalation).where(
                Escalation.conversation_id == payment.conversation_id,
                Escalation.agent_id == agent.id,
                Escalation.status == EscalationStatus.ASSIGNED,
            )
        )
    ).scalar_one_or_none()
    if escalation is None:
        raise HTTPException(
            status_code=409,
            detail="Claim this conversation before sending its e-ticket",
        )
    if payment.status.value != "success":
        raise HTTPException(
            status_code=409, detail="Payment has not succeeded yet"
        )

    ticket = await booking.issue_ticket(payment.pnr)
    if not ticket.get("ticketed"):
        raise HTTPException(
            status_code=409,
            detail=ticket.get("error") or "The e-ticket could not be issued",
        )

    message = Message(
        conversation_id=payment.conversation_id,
        sender=MessageSender.AGENT,
        content=booking.ticket_message(ticket),
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    await conversation_hub.send(
        payment.conversation_id,
        {"type": "agent_message", "message": message_to_dict(message)},
    )
    # Ticketing flips the booking to TICKETED and records the e-ticket number.
    # Push the refreshed payment/booking so open cards update in place.
    await payments.broadcast_payment_update(payment.id)

    payload = await payments.get_payment_payload(payment.id)
    return SendTicketOut(
        payment=PaymentOut(**payload),
        ticket=ticket,
        message=MessageOut.model_validate(message),
    )
