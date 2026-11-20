"""Start an M-Pesa STK push from the in-chat payment card."""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Booking, Payment, PaymentStatus
from services import mpesa

from .polling import schedule_poller
from .state import (
    PaymentError,
    _as_uuid,
    _broadcast,
    payment_payload,
)


async def start_payment(payment_id: str | uuid.UUID, raw_phone: str) -> dict:
    """Validate the number and send an STK push; returns the updated payment.

    A previously failed/cancelled/timed-out attempt is reset so the passenger
    can retry the same card. A successful payment is returned unchanged.
    """
    try:
        phone = mpesa.normalize_phone(raw_phone)
    except ValueError as exc:
        raise PaymentError(str(exc)) from exc

    payment_uuid = _as_uuid(payment_id)
    if payment_uuid is None:
        raise PaymentError("Payment not found.")

    async with SessionLocal() as db:
        payment = (
            await db.execute(select(Payment).where(Payment.id == payment_uuid))
        ).scalar_one_or_none()
        if payment is None:
            raise PaymentError("Payment not found.")
        booking = (
            await db.execute(select(Booking).where(Booking.pnr == payment.pnr))
        ).scalar_one_or_none()
        if payment.status == PaymentStatus.SUCCESS:
            return payment_payload(payment, booking)
        if payment.status not in (PaymentStatus.PENDING, PaymentStatus.FAILED,
                                  PaymentStatus.CANCELLED, PaymentStatus.TIMEOUT):
            raise PaymentError("This payment cannot be started.")

        # Reset a previous attempt before pushing again.
        payment.status = PaymentStatus.PENDING
        payment.receipt = None
        payment.checkout_request_id = None
        payment.merchant_request_id = None
        payment.raw = None
        payment.phone = phone
        await db.flush()

        try:
            response = await mpesa.get_client().stk_push(
                phone,
                float(payment.amount),
                payment.pnr,
                f"Kenya Airways {payment.pnr}",
            )
        except mpesa.MpesaError as exc:
            payment.status = PaymentStatus.FAILED
            payment.raw = {"error": str(exc)}
            await db.commit()
            await db.refresh(payment)
            payload = payment_payload(
                payment,
                booking,
                message=f"Could not send the M-Pesa prompt: {exc}",
            )
            await _broadcast(payment, booking, payload["message"])
            return payload

        payment.checkout_request_id = response.get("checkout_request_id")
        payment.merchant_request_id = response.get("merchant_request_id")
        payment.raw = response.get("raw")
        await db.commit()
        await db.refresh(payment)
        payload = payment_payload(payment, booking)
        await _broadcast(payment, booking)

    if payment.checkout_request_id:
        schedule_poller(payment.id, payment.checkout_request_id)
    return payload
