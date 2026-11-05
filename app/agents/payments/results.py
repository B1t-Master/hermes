"""Apply terminal M-Pesa results exactly once and fan out side effects."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select

from app.agents.hub import conversation_hub
from app.db import SessionLocal
from app.models import (
    Booking,
    BookingStatus,
    Conversation,
    ConversationStatus,
    Message,
    MessageSender,
    Payment,
    PaymentStatus,
)
from services import mpesa

from .state import (
    _as_uuid,
    _broadcast,
    _status_message,
    get_payment_payload,
    payment_payload,
)

logger = logging.getLogger(__name__)


async def apply_payment_result_by_checkout(
    checkout_request_id: str,
    *,
    status: str,
    receipt: str | None = None,
    result_code: str | None = None,
    result_desc: str | None = None,
    raw: dict | None = None,
) -> dict | None:
    """Resolve a payment by Daraja checkout id (used by the callback route)."""
    async with SessionLocal() as db:
        payment = (
            await db.execute(
                select(Payment).where(
                    Payment.checkout_request_id == checkout_request_id
                )
            )
        ).scalar_one_or_none()
    if payment is None:
        logger.warning("Callback for unknown checkout=%s", checkout_request_id)
        return None
    return await apply_payment_result(
        payment.id,
        status=status,
        receipt=receipt,
        result_code=result_code,
        result_desc=result_desc,
        raw=raw,
    )


async def apply_payment_result(
    payment_id: str | uuid.UUID,
    *,
    status: str,
    receipt: str | None = None,
    result_code: str | None = None,
    result_desc: str | None = None,
    raw: dict | None = None,
) -> dict | None:
    """Apply a terminal provider result exactly once and fan it out.

    Idempotent: a second callback/poll for an already-resolved payment returns
    the stored state without creating another escalation or message.
    """
    payment_uuid = _as_uuid(payment_id)
    if payment_uuid is None:
        return None

    try:
        new_status = PaymentStatus(status)
    except ValueError:
        logger.warning("Unknown payment status %r; treating as failed", status)
        new_status = PaymentStatus.FAILED

    async with SessionLocal() as db:
        payment = (
            await db.execute(select(Payment).where(Payment.id == payment_uuid))
        ).scalar_one_or_none()
        if payment is None:
            return None
        booking = (
            await db.execute(select(Booking).where(Booking.pnr == payment.pnr))
        ).scalar_one_or_none()

        if payment.status != PaymentStatus.PENDING:
            # Safaricom can deliver a success callback after our poll budget
            # expired. Let that late success win over the timeout, but never
            # reopen a payment that already failed, was cancelled, or succeeded.
            late_success = (
                new_status == PaymentStatus.SUCCESS
                and payment.status == PaymentStatus.TIMEOUT
            )
            if not late_success:
                return payment_payload(payment, booking)

        payment.status = new_status
        if receipt:
            payment.receipt = receipt
        if raw is not None:
            merged = dict(payment.raw or {})
            merged.update(raw)
            payment.raw = merged
        if result_code is not None or result_desc:
            merged = dict(payment.raw or {})
            merged["result_code"] = result_code
            merged["result_desc"] = result_desc
            payment.raw = merged

        if new_status == PaymentStatus.SUCCESS:
            if booking is not None and booking.status == BookingStatus.HELD:
                booking.status = BookingStatus.PAID

        await db.commit()
        await db.refresh(payment)
        if booking is not None:
            await db.refresh(booking)
        payload = payment_payload(
            payment,
            booking,
            message=mpesa.result_message(result_code, result_desc)
            if result_code is not None
            else _status_message(payment),
        )

    # Side effects happen outside the transaction so a broadcast failure never
    # rolls back the payment.
    if new_status == PaymentStatus.SUCCESS:
        await _on_payment_success(payment_uuid, receipt)
    else:
        await _broadcast(payment, booking, payload["message"])

    refreshed = await get_payment_payload(payment_uuid)
    return refreshed or payload


async def _on_payment_success(payment_id: uuid.UUID, receipt: str | None) -> None:
    """Mark the payment as successful and keep the conversation open.

    The conversation stays active so the passenger can provide their email
    address through chat for e-ticket delivery. No human escalation is created
    here — the bot handles email collection and ticket issuance directly.
    """
    async with SessionLocal() as db:
        payment = (
            await db.execute(select(Payment).where(Payment.id == payment_id))
        ).scalar_one_or_none()
        if payment is None or payment.conversation_id is None:
            return
        booking = (
            await db.execute(select(Booking).where(Booking.pnr == payment.pnr))
        ).scalar_one_or_none()
        conversation = (
            await db.execute(
                select(Conversation).where(Conversation.id == payment.conversation_id)
            )
        ).scalar_one_or_none()
        if conversation is None:
            return

        # Keep the conversation active so the bot can collect the email
        # and issue the ticket directly. No human escalation is created.
        conversation.status = ConversationStatus.ACTIVE
        conversation.escalation_reason = None

        receipt_value = receipt or payment.receipt
        if receipt_value:
            text = (
                f"Payment received — M-Pesa receipt {receipt_value}. "
                f"Booking reference: {payment.pnr}."
            )
        else:
            text = (
                f"Payment received. Booking reference: {payment.pnr}."
            )
        if booking is not None and not booking.passenger_email:
            facts = dict(conversation.facts or {})
            facts["needs_email"] = True
            facts["pending_email_booking"] = {
                "pnr": booking.pnr,
                "flight_no": booking.flight_no,
                "payment_id": str(payment.id),
            }
            conversation.facts = facts
        elif booking is not None and booking.passenger_email:
            try:
                ticket = await booking.issue_ticket(
                    booking.pnr, email=booking.passenger_email
                )
                eticket = ticket.get("eticket_number") or ""
                if ticket.get("ticketed"):
                    text += (
                        f"\nE-ticket {eticket} has been issued and emailed to "
                        f"{booking.passenger_email}."
                    )
                else:
                    note = ticket.get("email_note") or ""
                    text += "\n" + (note or "Your e-ticket will be issued shortly.")
            except Exception:
                logger.exception("Auto-issue failed for %s", booking.pnr)
            try:
                await booking.decrement_seats(booking.flight_no, booking.depart_date)
            except Exception:
                logger.exception("Seat decrement failed for %s", booking.flight_no)
        message = Message(
            conversation_id=conversation.id,
            sender=MessageSender.BOT,
            content=text,
        )
        db.add(message)
        await db.commit()
        await db.refresh(message)
        await db.refresh(payment)
        if booking is not None:
            await db.refresh(booking)
        payload = payment_payload(payment, booking)
        message_dict = {
            "id": str(message.id),
            "conversation_id": str(message.conversation_id),
            "sender": message.sender.value,
            "content": message.content,
            "sentiment_score": None,
            "sentiment_label": None,
            "sources": [],
            "actions": None,
            "payment_id": str(message.payment_id) if message.payment_id else None,
            "created_at": message.created_at.isoformat()
            if message.created_at
            else None,
        }

    try:
        await conversation_hub.send(
            payment.conversation_id,
            {"type": "payment_update", "payment": payload},
        )
        await conversation_hub.send(
            payment.conversation_id,
            {"type": "bot_message", "message": message_dict},
        )
    except Exception:
        logger.exception("Payment success fan-out failed for %s", payment_id)
