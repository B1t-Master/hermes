import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import payments
from app.agents.turns import run_turn
from app.db import get_db
from app.deps import get_current_passenger
from app.metrics import CSAT, add_metric
from app.models import (
    Conversation,
    ConversationStatus,
    Message,
    Passenger,
    Payment,
    PaymentStatus,
    PerformanceMetric,
)
from app.schemas import (
    ConversationOut,
    CSATIn,
    MessageIn,
    MessageOut,
    PaymentEmailIn,
    PaymentOut,
    PaymentPhoneIn,
    TurnOut,
)
from services import booking

router = APIRouter(prefix="/chat", tags=["chat"])

logger = logging.getLogger(__name__)


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    passenger: Passenger = Depends(get_current_passenger),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.passenger_id == passenger.id)
        .order_by(Conversation.created_at.desc())
    )
    return result.scalars().all()


@router.post("/conversations", response_model=ConversationOut, status_code=201)
async def create_conversation(
    passenger: Passenger = Depends(get_current_passenger),
    db: AsyncSession = Depends(get_db),
):
    conversation = Conversation(passenger_id=passenger.id)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation_id: uuid.UUID,
    passenger: Passenger = Depends(get_current_passenger),
    db: AsyncSession = Depends(get_db),
):
    conversation = await _get_owned_conversation(db, conversation_id, passenger.id)
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc())
    )
    return result.scalars().all()


@router.post("/conversations/{conversation_id}/messages", response_model=TurnOut)
async def send_message(
    conversation_id: uuid.UUID,
    payload: MessageIn,
    passenger: Passenger = Depends(get_current_passenger),
    db: AsyncSession = Depends(get_db),
):
    """Non-streaming fallback for the WS turn: same pipeline, one JSON reply."""
    conversation = await _get_owned_conversation(db, conversation_id, passenger.id)
    if conversation.status == ConversationStatus.RESOLVED:
        raise HTTPException(status_code=400, detail="Conversation is closed")

    try:
        result = await run_turn(conversation.id, payload.content)
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except LookupError:
        raise HTTPException(status_code=404, detail="Conversation not found") from None

    return TurnOut(
        passenger_message=result["passenger_message"],
        bot_message=result.get("bot_message"),
        escalate=bool(result.get("escalate")),
        escalation_reason=result.get("escalation_reason") or None,
        offer_escalation=bool(result.get("offer_escalation")),
        escalation_offer=result.get("escalation_offer"),
        intent=result.get("intent"),
        sentiment_score=result.get("sentiment_score"),
        sentiment_label=result.get("sentiment_label"),
    )


@router.get("/payments/{payment_id}", response_model=PaymentOut)
async def get_payment(
    payment_id: uuid.UUID,
    passenger: Passenger = Depends(get_current_passenger),
    db: AsyncSession = Depends(get_db),
):
    """Current state of an in-chat payment (polled by the payment card)."""
    payment = await _get_owned_payment(db, payment_id, passenger.id)
    payload = await payments.get_payment_payload(payment.id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Payment not found")
    return PaymentOut(**payload)


@router.post("/payments/{payment_id}/phone", response_model=PaymentOut)
async def submit_payment_phone(
    payment_id: uuid.UUID,
    payload: PaymentPhoneIn,
    passenger: Passenger = Depends(get_current_passenger),
    db: AsyncSession = Depends(get_db),
):
    """Send an M-Pesa STK push for a booking's payment card.

    The number travels browser -> backend only. It is stored on the payment
    row (masked in every response) and never enters the chat/LLM pipeline.
    """
    await _get_owned_payment(db, payment_id, passenger.id)
    try:
        result = await payments.start_payment(payment_id, payload.phone)
    except payments.PaymentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return PaymentOut(**result)


@router.post("/payments/{payment_id}/email", response_model=PaymentOut)
async def submit_payment_email(
    payment_id: uuid.UUID,
    payload: PaymentEmailIn,
    passenger: Passenger = Depends(get_current_passenger),
    db: AsyncSession = Depends(get_db),
):
    """Store an optional e-ticket email for a paid booking.

    The payment card asks for this after success. It does not issue the ticket;
    the agent dashboard sends it, and a copy goes to this address when set.
    """
    payment = await _get_owned_payment(db, payment_id, passenger.id)
    if payment.status != PaymentStatus.SUCCESS:
        raise HTTPException(status_code=409, detail="Payment has not succeeded yet")

    result = await booking.set_booking_email(payment.pnr, str(payload.email))
    if not result.get("updated"):
        raise HTTPException(
            status_code=404, detail=result.get("error") or "Booking not found"
        )

    # If the payment succeeded, issue the e-ticket and email it now so the
    # passenger receives it immediately (not only when an agent acts).
    ticket_result = None
    if payment.status == PaymentStatus.SUCCESS:
        try:
            ticket_result = await booking.issue_ticket(payment.pnr, email=str(payload.email))
        except Exception:
            logger.exception("Auto-issue failed for %s", payment.pnr)

    email_sent = False
    eticket = result.get("eticket_number")
    email_note = result.get("email_note", "")
    if ticket_result:
        eticket = ticket_result.get("eticket_number") or eticket
        email_sent = ticket_result.get("email_sent", False)
        email_note = ticket_result.get("email_note", "") or email_note

    # Notify the passenger in chat that the booking is confirmed and the
    # e-ticket has been sent (or will be, once payment is verified).
    origin = result.get("origin") or ""
    destination = result.get("destination") or ""
    route_label = f"{origin} → {destination}" if origin and destination else ""
    date_label = result.get("depart_date") or ""
    itinerary = (
        f"| Flight | Route | Date | Departs | Fare |\n"
        f"|--------|-------|------|---------|------|\n"
        f"| {result.get('flight_no') or ''} | {route_label} | "
        f"{date_label} | {result.get('depart_time') or ''} | "
        f"{result.get('currency') or 'KES'} {float(result.get('amount') or 0):,.0f} |\n"
    )
    if email_sent:
        msg_text = (
            f"Thank you for booking the flight! Your e-ticket "
            f"{'#' + eticket + ' ' if eticket else ''}"
            f"has been sent to {payload.email}.\n\n"
            f"Here's your booking:\n\n{itinerary}\n"
            f"Booking reference: {payment.pnr}."
        )
    else:
        msg_text = (
            f"Thank you for booking the flight! Your email has been saved. "
            f"{email_note or 'Your e-ticket will be issued shortly.'}\n\n"
            f"Here's your booking:\n\n{itinerary}\n"
            f"Booking reference: {payment.pnr}."
        )
    try:
        from app.models import Conversation, Message, MessageSender
        from app.db import SessionLocal
        async with SessionLocal() as msg_db:
            msg = Message(
                conversation_id=payment.conversation_id,
                sender=MessageSender.BOT,
                content=msg_text,
            )
            msg_db.add(msg)
            await msg_db.commit()
            await msg_db.refresh(msg)
            msg_dict = {
                "id": str(msg.id),
                "conversation_id": str(msg.conversation_id),
                "sender": "bot",
                "content": msg.content,
                "sentiment_score": None,
                "sentiment_label": None,
                "sources": [],
                "actions": None,
                "payment_id": None,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
        from app.agents.hub import conversation_hub
        await conversation_hub.send(
            payment.conversation_id,
            {"type": "bot_message", "message": msg_dict},
        )
    except Exception:
        logger.exception("Could not send thank-you message for %s", payment.pnr)

    updated = await payments.broadcast_payment_update(payment.id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Payment not found")
    return PaymentOut(**updated)


@router.post("/conversations/{conversation_id}/csat", status_code=204)
async def rate_conversation(
    conversation_id: uuid.UUID,
    payload: CSATIn,
    passenger: Passenger = Depends(get_current_passenger),
    db: AsyncSession = Depends(get_db),
):
    """End-of-conversation star rating (1..5). Resubmission replaces the value."""
    conversation = await _get_owned_conversation(db, conversation_id, passenger.id)
    await db.execute(
        delete(PerformanceMetric).where(
            PerformanceMetric.conversation_id == conversation.id,
            PerformanceMetric.metric == CSAT,
        )
    )
    add_metric(db, conversation.id, CSAT, payload.rating)
    await db.commit()


async def _get_owned_conversation(
    db: AsyncSession, conversation_id: uuid.UUID, passenger_id: uuid.UUID
) -> Conversation:
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.passenger_id == passenger_id
        )
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


async def _get_owned_payment(
    db: AsyncSession, payment_id: uuid.UUID, passenger_id: uuid.UUID
) -> Payment:
    """A payment is visible only through a conversation the passenger owns."""
    payment = (
        await db.execute(select(Payment).where(Payment.id == payment_id))
    ).scalar_one_or_none()
    if payment is None or payment.conversation_id is None:
        raise HTTPException(status_code=404, detail="Payment not found")
    conversation = (
        await db.execute(
            select(Conversation).where(
                Conversation.id == payment.conversation_id,
                Conversation.passenger_id == passenger_id,
            )
        )
    ).scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=404, detail="Payment not found")
    return payment