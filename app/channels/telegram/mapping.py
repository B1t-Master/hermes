"""DB mapping: Telegram chat -> passenger -> conversation."""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Conversation, ConversationStatus, Passenger


async def _get_or_create_passenger(chat_id: int, name: str) -> uuid.UUID:
    key = str(chat_id)
    async with SessionLocal() as db:
        existing = (
            await db.execute(select(Passenger).where(Passenger.telegram_chat_id == key))
        ).scalar_one_or_none()
        if existing is not None:
            return existing.id
        passenger = Passenger(
            first_name=(name or "Telegram user")[:120],
            is_anonymous=True,
            telegram_chat_id=key,
        )
        db.add(passenger)
        await db.commit()
        await db.refresh(passenger)
        return passenger.id


async def _get_or_create_conversation(passenger_id: uuid.UUID) -> uuid.UUID:
    """Reuse the passenger's current conversation, else open one.

    A resolved conversation is abandoned (the bot refuses turns on it); an
    escalated one is reused so the human agent keeps the same thread.
    """
    async with SessionLocal() as db:
        existing = (
            await db.execute(
                select(Conversation)
                .where(
                    Conversation.passenger_id == passenger_id,
                    Conversation.status != ConversationStatus.RESOLVED,
                )
                .order_by(Conversation.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing.id
        conversation = Conversation(passenger_id=passenger_id)
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        return conversation.id
