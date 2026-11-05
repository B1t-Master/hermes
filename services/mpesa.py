"""Safaricom Daraja (M-Pesa) STK push client with a simulated provider.

The real client talks to the sandbox/production Daraja API. The simulated
client resolves after a short delay and is used when credentials are missing,
so local runs and tests can exercise the whole booking flow without a phone.

Only the server ever sees an M-Pesa number: the browser posts it straight to
the payment endpoint, and the number is never part of an LLM prompt.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Protocol

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

SANDBOX_BASE = "https://sandbox.safaricom.co.ke"
PRODUCTION_BASE = "https://api.safaricom.co.ke"

# Daraja result codes we care about.
RESULT_SUCCESS = "0"
RESULT_CANCELLED = "1032"
RESULT_TIMEOUT = "1037"
RESULT_EXPIRED = "1019"
RESULT_INSUFFICIENT_FUNDS = "1"
RESULT_WRONG_PIN = "2001"
RESULT_LOCKED = "1001"


class MpesaError(RuntimeError):
    """A Daraja request failed or the payload was unusable."""


class PaymentProvider(Protocol):
    name: str

    async def stk_push(
        self,
        phone: str,
        amount: float,
        account_reference: str,
        description: str,
    ) -> dict: ...

    async def query(self, checkout_request_id: str) -> dict: ...


def normalize_phone(raw: str | None) -> str:
    """Normalise a Kenyan mobile number to Daraja's ``2547XXXXXXXX`` form.

    Accepts ``0712 345 678``, ``+254712345678``, ``254712345678`` and
    ``712345678``. Raises ``ValueError`` for anything that is not a Kenyan
    mobile number.
    """
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("254"):
        pass
    elif digits.startswith("0"):
        digits = "254" + digits[1:]
    elif digits.startswith(("7", "1")):
        digits = "254" + digits
    else:
        raise ValueError("Enter a Kenyan mobile number, for example 0712 345 678.")
    if not re.fullmatch(r"254[17]\d{8}", digits):
        raise ValueError("Enter a valid Kenyan mobile number, for example 0712 345 678.")
    return digits


def mask_phone(phone: str | None) -> str:
    """``254712345678`` -> ``0712 *** 678`` for UI display."""
    if not phone:
        return ""
    digits = re.sub(r"\D", "", phone)
    if re.fullmatch(r"254[17]\d{8}", digits):
        local = "0" + digits[3:]
        return f"{local[:4]} *** {local[-3:]}"
    return phone


def result_status(result_code: str | int | None) -> str:
    """Map a Daraja result code to a Payment status value.

    ``pending`` means the caller should keep polling. ``timeout`` is returned
    only after the poll budget is exhausted, so 1037/1001 are treated as
    pending here.
    """
    code = str(result_code if result_code is not None else "").strip()
    if code == RESULT_SUCCESS:
        return "success"
    if code == RESULT_CANCELLED:
        return "cancelled"
    if code == RESULT_EXPIRED:
        return "timeout"
    if code in (RESULT_TIMEOUT, RESULT_LOCKED, "") or code == "4999":
        return "pending"
    return "failed"


def result_message(result_code: str | int | None, result_desc: str | None = None) -> str:
    """Passenger-friendly explanation for a Daraja result."""
    code = str(result_code if result_code is not None else "").strip()
    if code == RESULT_SUCCESS:
        return "Payment received."
    if code == RESULT_CANCELLED:
        return "The M-Pesa prompt was cancelled."
    if code == RESULT_EXPIRED:
        return "The M-Pesa prompt expired. Please try again."
    if code == RESULT_TIMEOUT:
        return "The M-Pesa prompt timed out. Please try again."
    if code == RESULT_LOCKED:
        return "Another M-Pesa transaction is already in progress on this number."
    if code == RESULT_INSUFFICIENT_FUNDS:
        return "The M-Pesa balance is too low to complete this payment."
    if code == RESULT_WRONG_PIN:
        return "The M-Pesa PIN was entered incorrectly."
    return result_desc or "The payment was not completed."


def extract_metadata(callback: dict) -> dict:
    """Pull useful values out of a Daraja CallbackMetadata block."""
    items = (
        (callback.get("CallbackMetadata") or {}).get("Item")
        if isinstance(callback.get("CallbackMetadata"), dict)
        else None
    ) or []
    values: dict[str, object] = {}
    for item in items:
        if isinstance(item, dict) and item.get("Name"):
            values[str(item["Name"])] = item.get("Value")
    return values


class DarajaClient:
    """Real Daraja STK push + query client."""

    name = "daraja"

    def __init__(self) -> None:
        self._base = SANDBOX_BASE if settings.mpesa_env != "production" else PRODUCTION_BASE
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()

    async def _access_token(self) -> str:
        now = time.monotonic()
        if self._token and now < self._token_expires_at:
            return self._token
        async with self._token_lock:
            now = time.monotonic()
            if self._token and now < self._token_expires_at:
                return self._token
            try:
                async with httpx.AsyncClient(timeout=settings.mpesa_timeout_seconds) as client:
                    response = await client.get(
                        f"{self._base}/oauth/v1/generate",
                        params={"grant_type": "client_credentials"},
                        auth=(
                            settings.mpesa_consumer_key,
                            settings.mpesa_consumer_secret,
                        ),
                    )
            except httpx.HTTPError as exc:
                raise MpesaError(f"Could not reach M-Pesa: {exc}") from exc
            if response.status_code >= 400:
                raise MpesaError(
                    f"M-Pesa authentication failed ({response.status_code}): {response.text[:200]}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise MpesaError("M-Pesa returned an unreadable authentication response") from exc
            token = payload.get("access_token")
            if not token:
                raise MpesaError("M-Pesa authentication returned no access token")
            expires_in = int(payload.get("expires_in") or 3599)
            self._token = str(token)
            # Refresh a minute early so a long request never races expiry.
            self._token_expires_at = time.monotonic() + max(30, expires_in - 60)
            return self._token

    @staticmethod
    def _password(timestamp: str) -> str:
        raw = f"{settings.mpesa_shortcode}{settings.mpesa_passkey}{timestamp}"
        return base64.b64encode(raw.encode()).decode()

    async def _post(self, path: str, payload: dict) -> dict:
        token = await self._access_token()
        try:
            async with httpx.AsyncClient(timeout=settings.mpesa_timeout_seconds) as client:
                response = await client.post(
                    f"{self._base}{path}",
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                )
        except httpx.HTTPError as exc:
            raise MpesaError(f"Could not reach M-Pesa: {exc}") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise MpesaError(
                f"M-Pesa returned an unreadable response ({response.status_code})"
            ) from exc
        if response.status_code >= 400 or (
            isinstance(data, dict) and data.get("errorCode")
        ):
            detail = ""
            if isinstance(data, dict):
                detail = (
                    data.get("errorMessage")
                    or data.get("ResponseDescription")
                    or data.get("errorCode")
                    or ""
                )
            raise MpesaError(f"M-Pesa request failed: {detail or response.status_code}")
        return data

    async def stk_push(
        self,
        phone: str,
        amount: float,
        account_reference: str,
        description: str,
    ) -> dict:
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        payload = {
            "BusinessShortCode": settings.mpesa_shortcode,
            "Password": self._password(timestamp),
            "Timestamp": timestamp,
            "TransactionType": "CustomerPayBillOnline",
            "Amount": int(round(amount)),  # noqa: RUF046
            "PartyA": phone,
            "PartyB": settings.mpesa_shortcode,
            "PhoneNumber": phone,
            "CallBackURL": settings.mpesa_callback,
            "AccountReference": (account_reference or "KQ")[:12],
            "TransactionDesc": (description or "Kenya Airways")[:20],
        }
        data = await self._post("/mpesa/stkpush/v1/processrequest", payload)
        if str(data.get("ResponseCode", "")) != "0":
            raise MpesaError(
                data.get("ResponseDescription")
                or data.get("errorMessage")
                or "M-Pesa rejected the STK push"
            )
        return {
            "provider": self.name,
            "merchant_request_id": data.get("MerchantRequestID"),
            "checkout_request_id": data.get("CheckoutRequestID"),
            "response_code": str(data.get("ResponseCode", "")),
            "response_description": data.get("ResponseDescription"),
            "customer_message": data.get("CustomerMessage"),
            "raw": data,
        }

    async def query(self, checkout_request_id: str) -> dict:
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        payload = {
            "BusinessShortCode": settings.mpesa_shortcode,
            "Password": self._password(timestamp),
            "Timestamp": timestamp,
            "CheckoutRequestID": checkout_request_id,
        }
        data = await self._post("/mpesa/stkpushquery/v1/query", payload)
        result_code = data.get("ResultCode")
        return {
            "provider": self.name,
            "result_code": str(result_code) if result_code is not None else None,
            "result_desc": data.get("ResultDesc"),
            "status": result_status(result_code),
            "message": result_message(result_code, data.get("ResultDesc")),
            "raw": data,
        }


class SimulatedClient:
    """Deterministic stand-in used when Daraja credentials are unavailable.

    A fake STK push resolves after ``MPESA_SIMULATE_DELAY_SECONDS``. Phones
    ending in ``0000`` fail (insufficient funds) and ``1111`` cancel, which
    keeps the failure paths demonstrable without a real handset.
    """

    name = "simulated"

    def __init__(self) -> None:
        self._started: dict[str, tuple[float, str]] = {}

    async def stk_push(
        self,
        phone: str,
        amount: float,
        account_reference: str,
        description: str,
    ) -> dict:
        checkout_id = f"ws_CO_SIM_{uuid.uuid4().hex[:16].upper()}"
        merchant_id = f"SIM-{uuid.uuid4().hex[:12].upper()}"
        self._started[checkout_id] = (time.monotonic(), phone)
        logger.info(
            "Simulated STK push checkout=%s phone=%s amount=%s reference=%s",
            checkout_id,
            phone,
            amount,
            account_reference,
        )
        return {
            "provider": self.name,
            "merchant_request_id": merchant_id,
            "checkout_request_id": checkout_id,
            "response_code": "0",
            "response_description": "Success. Request accepted for processing",
            "customer_message": (
                "A simulated M-Pesa prompt was sent. It resolves automatically "
                f"after {settings.mpesa_simulate_delay_seconds:.0f}s."
            ),
            "raw": {"simulated": True, "phone": phone},
        }

    async def query(self, checkout_request_id: str) -> dict:
        started = self._started.get(checkout_request_id)
        if started is None:
            return {
                "provider": self.name,
                "result_code": RESULT_TIMEOUT,
                "result_desc": "Unknown checkout request",
                "status": "failed",
                "message": "The payment session could not be found.",
                "raw": {"simulated": True},
            }
        elapsed = time.monotonic() - started[0]
        phone = started[1]
        if elapsed < settings.mpesa_simulate_delay_seconds:
            return {
                "provider": self.name,
                "result_code": None,
                "result_desc": "Pending",
                "status": "pending",
                "message": "Waiting for the M-Pesa prompt to be completed.",
                "raw": {"simulated": True, "elapsed": elapsed},
            }
        if phone.endswith("0000"):
            code, desc = RESULT_INSUFFICIENT_FUNDS, "Insufficient funds"
        elif phone.endswith("1111"):
            code, desc = RESULT_CANCELLED, "Request cancelled by user"
        else:
            code, desc = RESULT_SUCCESS, "The service request is processed successfully."
        receipt = None
        if code == RESULT_SUCCESS:
            digest = hashlib.sha256(f"{checkout_request_id}:{phone}".encode()).hexdigest()
            receipt = f"SIM{digest[:6].upper()}"
        return {
            "provider": self.name,
            "result_code": code,
            "result_desc": desc,
            "status": result_status(code),
            "message": result_message(code, desc),
            "receipt": receipt,
            "raw": {"simulated": True, "elapsed": elapsed},
        }


_client: PaymentProvider | None = None


def get_client() -> PaymentProvider:
    """Process-wide payment provider selected by configuration."""
    global _client
    if _client is None:
        if settings.mpesa_effective_provider == "daraja":
            _client = DarajaClient()
        else:
            logger.warning(
                "M-Pesa credentials are not configured; using the simulated provider"
            )
            _client = SimulatedClient()
    return _client


def reset_client() -> None:
    """Test hook: forget the cached provider so settings changes take effect."""
    global _client
    _client = None
