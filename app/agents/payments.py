"""M-Pesa payment orchestration for the in-chat booking flow.

The passenger enters their number in a payment card, the browser posts it to
``POST /chat/payments/{id}/phone``, and this module sends the STK push. A short
background poller watches Daraja (and the simulated provider) until the payment
resolves. On success the booking is marked paid, a human escalation is created,
the passenger gets a bot message, and the dashboard is told to update.

The number is only ever handled here and in the payment endpoints: it is never
part of an LLM prompt or a chat message.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from sqlalchemy import select

from app.agents.hub import conversation_hub
from app.config import settings
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

logger = logging.getLogger(__name__)

# payment id -> poller task, so a duplicate submit does not start two loops.
_pollers: dict[str, asyncio.Task] = {}


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


def schedule_poller(payment_id: str | uuid.UUID, checkout_request_id: str) -> None:
    """Start (or reuse) one background poller for a checkout request."""
    key = str(payment_id)
    existing = _pollers.get(key)
    if existing is not None and not existing.done():
        return
    try:
        _pollers[key] = asyncio.create_task(
            _poll_payment(payment_id, checkout_request_id)
        )
    except RuntimeError:
        # No running loop (e.g. a unit test calling start_payment directly).
        logger.debug("No event loop; payment poller not scheduled for %s", payment_id)


async def _poll_payment(payment_id: str | uuid.UUID, checkout_request_id: str) -> None:
    """Poll the provider until the payment reaches a terminal state."""
    interval = max(0.05, float(settings.mpesa_stk_query_interval_seconds))
    attempts = max(1, int(settings.mpesa_stk_query_max_attempts))
    for _ in range(attempts):
        await asyncio.sleep(interval)
        try:
            probe = await mpesa.get_client().query(checkout_request_id)
        except mpesa.MpesaError as exc:
            logger.warning("M-Pesa status query failed for %s: %s", checkout_request_id, exc)
            continue
        status = probe.get("status") or "pending"
        if status == "pending":
            continue
        await apply_payment_result(
            payment_id,
            status=status,
            receipt=probe.get("receipt"),
            result_code=probe.get("result_code"),
            result_desc=probe.get("result_desc"),
            raw=probe.get("raw"),
        )
        return

    logger.info("M-Pesa poll budget exhausted for checkout=%s", checkout_request_id)
    await apply_payment_result(
        payment_id,
        status=PaymentStatus.TIMEOUT.value,
        result_code=mpesa.RESULT_TIMEOUT,
        result_desc="No confirmation received from M-Pesa",
        raw={"poll_exhausted": True},
    )


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


async def resume_pending_payments() -> None:
    """Restart pollers for payments that were in flight at shutdown."""
    try:
        async with SessionLocal() as db:
            rows = (
                await db.execute(
                    select(Payment).where(
                        Payment.status == PaymentStatus.PENDING,
                        Payment.checkout_request_id.is_not(None),
                    )
                )
            ).scalars().all()
    except Exception:
        logger.exception("Could not resume pending M-Pesa payments")
        return
    for payment in rows:
        schedule_poller(payment.id, payment.checkout_request_id)
    if rows:
        logger.info("Resumed %d pending M-Pesa payment(s)", len(rows))


async def shutdown_payments() -> None:
    """Cancel pollers (application shutdown and tests)."""
    tasks = list(_pollers.values())
    _pollers.clear()
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
