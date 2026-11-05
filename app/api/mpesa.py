"""Safaricom Daraja callback endpoint.

Safaricom POSTs the STK result here after the passenger enters (or cancels)
their M-Pesa PIN. The route path deliberately avoids the substring ``mpesa``:
Safaricom rejects callback URLs containing it with ``400.003.02``.

The endpoint always answers ``200`` with the expected acknowledgement, even
when the payload is malformed or the checkout id is unknown: a non-2xx makes
Safaricom retry, and the poller covers any callback that never arrives.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from app.agents import payments
from app.config import settings
from services import mpesa

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mpesa"])

_ACCEPTED = {"ResultCode": 0, "ResultDesc": "Accepted"}


@router.post("/daraja/callback")
async def daraja_callback(
    request: Request,
    token: str | None = Query(default=None),
) -> dict:
    """Receive an STK push result and resolve the matching payment."""
    if settings.mpesa_callback_secret and token != settings.mpesa_callback_secret:
        raise HTTPException(status_code=403, detail="Invalid callback token")

    try:
        payload = await request.json()
    except ValueError:
        logger.warning("Daraja callback with unreadable JSON")
        return _ACCEPTED
    if not isinstance(payload, dict):
        logger.warning("Daraja callback payload was not an object")
        return _ACCEPTED

    callback = (payload.get("Body") or {}).get("stkCallback") or {}
    checkout_request_id = callback.get("CheckoutRequestID")
    result_code = callback.get("ResultCode")
    result_desc = callback.get("ResultDesc")
    metadata = mpesa.extract_metadata(callback)

    if not checkout_request_id:
        logger.warning("Daraja callback without CheckoutRequestID: %s", payload)
        return _ACCEPTED

    status = mpesa.result_status(result_code)
    if status == "pending":
        # 1037/1001 are still in progress; the poller keeps watching.
        return _ACCEPTED

    receipt = metadata.get("MpesaReceiptNumber")
    logger.info(
        "Daraja callback checkout=%s result=%s status=%s receipt=%s",
        checkout_request_id,
        result_code,
        status,
        receipt,
    )
    try:
        await payments.apply_payment_result_by_checkout(
            str(checkout_request_id),
            status=status,
            receipt=str(receipt) if receipt else None,
            result_code=str(result_code) if result_code is not None else None,
            result_desc=str(result_desc) if result_desc else None,
            raw={"callback": payload},
        )
    except Exception:
        # Record it, but still acknowledge so Safaricom does not hammer us;
        # the poller is the safety net.
        logger.exception("Applying Daraja callback failed")

    return _ACCEPTED
