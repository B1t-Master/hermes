"""Background polling for in-flight M-Pesa payments."""

from __future__ import annotations

import asyncio
import logging
import uuid

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import Payment, PaymentStatus
from services import mpesa

from .results import apply_payment_result

logger = logging.getLogger(__name__)

# payment id -> poller task, so a duplicate submit does not start two loops.
_pollers: dict[str, asyncio.Task] = {}


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
