"""Payment state: model loading, masking, status text, and card payloads."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select

from app.agents.hub import conversation_hub
from app.db import SessionLocal
from app.models import Booking, Payment, PaymentStatus
from services import mpesa

logger = logging.getLogger(__name__)


class PaymentError(RuntimeError):
    """The payment could not be started (bad number, unknown payment, ...)."""


def _as_uuid(value: str | uuid.UUID | None) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None


def _status_value(payment: Payment) -> str:
    return payment.status.value if payment.status else PaymentStatus.PENDING.value


def _status_message(payment: Payment) -> str:
    status = _status_value(payment)
    if status == PaymentStatus.PENDING.value:
        if payment.checkout_request_id:
            return (
                "A payment prompt was sent to your phone. Enter your M-Pesa PIN "
                "to complete the payment."
            )
        return "Enter your M-Pesa number to receive the payment prompt."
    if status == PaymentStatus.SUCCESS.value:
        return "Payment received. An agent is preparing your e-ticket."
    if status == PaymentStatus.CANCELLED.value:
        return "The M-Pesa prompt was cancelled. You can try again."
    if status == PaymentStatus.TIMEOUT.value:
        return "The M-Pesa prompt timed out. You can try again."
    if status == PaymentStatus.FAILED.value:
        return "The payment was not completed. You can try again."
    return ""


def payment_payload(
    payment: Payment,
    booking: Booking | None = None,
    *,
    message: str | None = None,
) -> dict:
    """Everything the payment card and dashboard need, as a plain dict."""
    payload = {
        "id": payment.id,
        "conversation_id": payment.conversation_id,
        "pnr": payment.pnr,
        "phone": payment.phone,
        "phone_masked": mpesa.mask_phone(payment.phone),
        "amount": payment.amount,
        "currency": payment.currency,
        "status": _status_value(payment),
        "receipt": payment.receipt,
        "checkout_request_id": payment.checkout_request_id,
        "message": message or _status_message(payment),
        "created_at": payment.created_at,
        "updated_at": payment.updated_at,
    }
    if booking is not None:
        payload.update(
            {
                "flight_no": booking.flight_no,
                "origin": booking.origin,
                "destination": booking.destination,
                "depart_date": booking.depart_date,
                "depart_time": booking.depart_time,
                "passenger_name": booking.passenger_name,
                "passenger_email": booking.passenger_email,
                "booking_status": booking.status.value if booking.status else None,
                "eticket_number": booking.eticket_number,
            }
        )
    return payload


async def load_payment(payment_id: str | uuid.UUID) -> tuple[Payment | None, Booking | None]:
    """Load a payment and its booking in one round trip each."""
    payment_uuid = _as_uuid(payment_id)
    if payment_uuid is None:
        return None, None
    async with SessionLocal() as db:
        payment = (
            await db.execute(select(Payment).where(Payment.id == payment_uuid))
        ).scalar_one_or_none()
        if payment is None:
            return None, None
        booking = (
            await db.execute(select(Booking).where(Booking.pnr == payment.pnr))
        ).scalar_one_or_none()
    return payment, booking


async def get_payment_payload(payment_id: str | uuid.UUID) -> dict | None:
    payment, booking = await load_payment(payment_id)
    if payment is None:
        return None
    return payment_payload(payment, booking)


async def broadcast_payment_update(payment_id: str | uuid.UUID) -> dict | None:
    """Push the current payment + booking state to the conversation.

    Used after ticketing changes the booking (status / e-ticket number) so any
    open payment card refreshes without a reload. Mutates nothing.
    """
    payment, booking = await load_payment(payment_id)
    if payment is None:
        return None
    await _broadcast(payment, booking)
    return payment_payload(payment, booking)


async def _broadcast(payment: Payment, booking: Booking | None, message: str | None = None) -> None:
    if payment.conversation_id is None:
        return
    try:
        await conversation_hub.send(
            payment.conversation_id,
            {
                "type": "payment_update",
                "payment": payment_payload(payment, booking, message=message),
            },
        )
    except Exception:
        logger.exception("Payment update broadcast failed for %s", payment.id)
