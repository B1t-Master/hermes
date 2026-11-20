"""Conversation persistence helpers used by the turn pipeline and API layer."""

from __future__ import annotations

import uuid

from sqlalchemy import func, or_, select

from app.db import SessionLocal
from app.models import Booking, Conversation, Message, MessageSender, Passenger

HISTORY_LIMIT = 10

_ROLE = {
    MessageSender.PASSENGER.value: "user",
    MessageSender.BOT.value: "assistant",
    MessageSender.AGENT.value: "assistant",
}


def message_to_dict(message: Message) -> dict:
    return {
        "id": str(message.id),
        "conversation_id": str(message.conversation_id),
        "sender": message.sender.value,
        "content": message.content,
        "sentiment_score": message.sentiment_score,
        "sentiment_label": message.sentiment_label,
        "sources": message.sources or [],
        "created_at": message.created_at.isoformat() if message.created_at else None,
        "actions": message.actions or None,
        "payment_id": str(message.payment_id) if message.payment_id else None,
    }


async def load_history(conversation_id: uuid.UUID, limit: int = HISTORY_LIMIT) -> list[dict]:
    """Prior turns as chat messages, oldest first."""
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
        ).scalars().all()
    rows = list(reversed(rows))
    return [
        {"role": _ROLE.get(m.sender.value, "assistant"), "content": m.content}
        for m in rows
    ]


async def get_conversation(conversation_id: uuid.UUID) -> Conversation | None:
    async with SessionLocal() as db:
        return (
            await db.execute(select(Conversation).where(Conversation.id == conversation_id))
        ).scalar_one_or_none()


def booking_to_dict(booking: Booking) -> dict:
    """A booking as a small, prompt-friendly dict."""
    return {
        "pnr": booking.pnr,
        "flight_no": booking.flight_no,
        "origin": booking.origin,
        "destination": booking.destination,
        "date": booking.depart_date,
        "time": booking.depart_time,
        "status": booking.status.value if booking.status else None,
    }


async def load_profile(passenger_id: uuid.UUID) -> dict:
    """Identity + bookings for a passenger, for prompt inheritance.

    Logged-in passengers carry their name and email; their bookings are matched
    by ``passenger_id`` or by the profile email, so a booking made on another
    channel (or seeded in the prototype) still shows up here. Anonymous
    passengers have neither, and only conversation facts are used.
    """
    async with SessionLocal() as db:
        passenger = (
            await db.execute(select(Passenger).where(Passenger.id == passenger_id))
        ).scalar_one_or_none()
        if passenger is None:
            return {}

        conditions = [Booking.passenger_id == passenger_id]
        if passenger.email:
            conditions.append(
                func.lower(Booking.passenger_email) == passenger.email.lower()
            )
        bookings = (
            await db.execute(
                select(Booking)
                .where(or_(*conditions))
                .order_by(Booking.depart_date.asc().nulls_last())
            )
        ).scalars().all()

    return {
        "name": " ".join(
            part for part in (passenger.first_name, passenger.last_name) if part
        ),
        "email": passenger.email,
        "anonymous": passenger.is_anonymous,
        "bookings": [booking_to_dict(booking) for booking in bookings],
    }
