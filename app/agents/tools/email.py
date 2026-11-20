"""Rebooking confirmation email delivery."""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import Booking
from services import email_real, rebook

logger = logging.getLogger(__name__)


async def _lookup_ticket_email(ticket_number: str | None) -> str | None:
    """Destination email for a ticket, read from the booking's stored email.

    Previously used ``RebookEmailMap``; now reads the single source of truth
    ``Booking.passenger_email`` which is the email captured at booking time and
    also used for e-ticket delivery.
    """
    ticket = (ticket_number or "").strip()
    if not ticket:
        return None
    async with SessionLocal() as db:
        booking = (
            await db.execute(select(Booking).where(Booking.pnr == ticket.upper()))
        ).scalar_one_or_none()
    if booking is None:
        return None
    return booking.passenger_email


async def _lookup_from_package(ticket_number: str | None) -> str | None:
    """Resolve the lookup through the package namespace.

    Keeps the historical ``app.agents.tools._lookup_ticket_email`` patch point
    working now that the implementation lives in a submodule.
    """
    from app.agents import tools as tools_pkg

    return await tools_pkg._lookup_ticket_email(ticket_number)


async def _send_rebook_confirmation(result: dict) -> None:
    """Email the confirmation when the ticket has an address on file.

    Reads the passenger email from the booking (the same email used for the
    e-ticket). The rebooking already succeeded, so delivery problems do not
    block the flow.
    """
    to = await _lookup_from_package(result.get("ticket_number"))
    if not to:
        result["email_sent"] = False
        result["email_note"] = (
            "No email address is on file for this booking reference, so no "
            "confirmation was sent."
        )
        return
    subject, body = rebook.confirmation_email(result)
    try:
        record = await asyncio.to_thread(
            email_real.send_email,
            to,
            subject,
            body,
            sender=settings.resend_from_email,
            api_key=settings.resend_api_key,
            outbox_dir=settings.email_outbox_dir,
        )
    except Exception:
        logger.exception("Rebook confirmation email failed for %s", result.get("reference"))
        return
    result["email_sent"] = True
    result["email_to"] = to
    result["email_provider"] = record.get("provider")
    result["email_sent_at"] = record.get("sent_at")
