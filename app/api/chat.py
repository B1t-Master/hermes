import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.turns import run_turn
from app.db import get_db
from app.deps import get_current_passenger
from app.models import Conversation, ConversationStatus, Message, Passenger
from app.schemas import ConversationOut, MessageIn, MessageOut, TurnOut

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    passenger: Passenger = Depends(get_current_passenger),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.passenger_id == passenger.id)
        .order_by(Conversation.created_at.desc())
    )
    return result.scalars().all()


@router.post("/conversations", response_model=ConversationOut, status_code=201)
async def create_conversation(
    passenger: Passenger = Depends(get_current_passenger),
    db: AsyncSession = Depends(get_db),
):
    conversation = Conversation(passenger_id=passenger.id)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation_id: uuid.UUID,
    passenger: Passenger = Depends(get_current_passenger),
    db: AsyncSession = Depends(get_db),
):
    conversation = await _get_owned_conversation(db, conversation_id, passenger.id)
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc())
    )
    return result.scalars().all()


@router.post("/conversations/{conversation_id}/messages", response_model=TurnOut)
async def send_message(
    conversation_id: uuid.UUID,
    payload: MessageIn,
    passenger: Passenger = Depends(get_current_passenger),
    db: AsyncSession = Depends(get_db),
):
    """Non-streaming fallback for the WS turn: same pipeline, one JSON reply."""
    conversation = await _get_owned_conversation(db, conversation_id, passenger.id)
    if conversation.status == ConversationStatus.RESOLVED:
        raise HTTPException(status_code=400, detail="Conversation is closed")

    try:
        result = await run_turn(conversation.id, payload.content)
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except LookupError:
        raise HTTPException(status_code=404, detail="Conversation not found") from None

    return TurnOut(
        passenger_message=result["passenger_message"],
        bot_message=result.get("bot_message"),
        escalate=bool(result.get("escalate")),
        escalation_reason=result.get("escalation_reason") or None,
        intent=result.get("intent"),
        sentiment_score=result.get("sentiment_score"),
        sentiment_label=result.get("sentiment_label"),
    )


async def _get_owned_conversation(
    db: AsyncSession, conversation_id: uuid.UUID, passenger_id: uuid.UUID
) -> Conversation:
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.passenger_id == passenger_id
        )
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation