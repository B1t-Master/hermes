"""WebSocket endpoints: one channel per conversation + one dashboard channel.

Channels
--------
- ``/ws/conversations/{id}?token=...``  Passenger (owner) or the agent who
  claimed the conversation's escalation. One room per conversation; everyone
  in the room receives the room broadcasts below.
- ``/ws/dashboard?token=...``           Agent dashboards. Receive ``queue_update``
  events whenever an escalation is created / claimed / released / resolved.

Envelope: every frame is a JSON object with a ``type`` field.

  client -> server
      user_message   {type, content}          passenger sends a turn
      agent_message  {type, content}          assigned agent replies
      typing         {type}                   relayed to the room
      ping / pong    {type}                   heartbeat

  server -> sender (unicast, turn-scoped)
      typing         {type, sender: "bot"}    bot is thinking
      bot_token      {type, content}          one streamed LLM token
      bot_done       {type, message}          final bot message persisted
      bot_error      {type, detail}           turn failed; nothing persisted
      error          {type, detail}           bad frame / permission problem

  server -> room (broadcast)
      passenger_message {type, message}        dedupe by id on the sender
      agent_message     {type, message}        dedupe by id on the sender
      bot_message       {type, message}        id appears first in bot_done
      typing            {type, sender}
      escalated         {type, reason}
      resolved          {type, escalation_id}

  server -> dashboards (broadcast)
      queue_update    {type, action, escalation_id, conversation_id}
                      action: created | claimed | released | resolved

Close codes: 4401 unauthenticated, 4403 forbidden, 4410 conversation closed.
"""

import asyncio
import logging
import uuid

import jwt
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.agents.hub import conversation_hub, dashboard_hub
from app.agents.turns import get_conversation, message_to_dict, run_turn
from app.db import SessionLocal
from app.models import Agent, Escalation, EscalationStatus, Message, MessageSender
from app.security import AGENT, PASSENGER, decode_access_token

logger = logging.getLogger(__name__)

router = APIRouter()

CLOSE_UNAUTHORIZED = 4401
CLOSE_FORBIDDEN = 4403
CLOSE_CLOSED = 4410
CLOSE_SERVER_ERROR = 1011

MAX_CONTENT_LENGTH = 4000
HEARTBEAT_SECONDS = 30

# One heartbeat task per accepted connection; cancelled on disconnect.
_heartbeats: set[asyncio.Task] = set()


def _decode_ws_token(token: str) -> tuple[str, uuid.UUID] | None:
    """JWT -> (role, subject id), or None when unusable."""
    try:
        payload = decode_access_token(token)
        role = payload["role"]
        subject = uuid.UUID(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError, TypeError):
        return None
    if role not in (PASSENGER, AGENT):
        return None
    return role, subject


async def _start_heartbeat(ws: WebSocket) -> None:
    task = asyncio.create_task(_heartbeat(ws))
    _heartbeats.add(task)
    task.add_done_callback(_heartbeats.discard)


async def _heartbeat(ws: WebSocket) -> None:
    try:
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            await ws.send_json({"type": "ping"})
    except Exception:
        return  # socket gone; receive loop will clean up the room


def _event(type_: str, **kwargs) -> dict:
    return {"type": type_, **kwargs}


# --------------------------------------------------------------------------
# Conversation channel
# --------------------------------------------------------------------------


@router.websocket("/ws/conversations/{conversation_id}")
async def conversation_socket(ws: WebSocket, conversation_id: uuid.UUID, token: str = Query(...)):
    auth = _decode_ws_token(token)
    if auth is None:
        await ws.close(code=CLOSE_UNAUTHORIZED, reason="invalid token")
        return
    role, subject = auth

    conversation = await get_conversation(conversation_id)
    if conversation is None:
        await ws.close(code=CLOSE_FORBIDDEN, reason="conversation not found")
        return

    if role == PASSENGER:
        if conversation.passenger_id != subject:
            await ws.close(code=CLOSE_FORBIDDEN, reason="not your conversation")
            return
    else:  # agent: only the assignee may enter
        async with SessionLocal() as db:
            escalation = (
                await db.execute(
                    select(Escalation).where(
                        Escalation.conversation_id == conversation_id,
                        Escalation.agent_id == subject,
                        Escalation.status == EscalationStatus.ASSIGNED,
                    )
                )
            ).scalar_one_or_none()
        if escalation is None:
            await ws.close(code=CLOSE_FORBIDDEN, reason="claim the escalation first")
            return

    from app.models import ConversationStatus

    if conversation.status == ConversationStatus.RESOLVED:
        await ws.close(code=CLOSE_CLOSED, reason="conversation is closed")
        return

    await ws.accept()
    conversation_hub.join(conversation_id, ws)
    await _start_heartbeat(ws)
    logger.info("WS conversation=%s role=%s connected", conversation_id, role)

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")
            if msg_type == "ping":
                await ws.send_json(_event("pong"))
            elif msg_type == "pong":
                continue
            elif msg_type == "typing":
                await conversation_hub.send(conversation_id, _event("typing", sender=role))
            elif msg_type == "user_message" and role == PASSENGER:
                await _handle_passenger_turn(ws, conversation_id, data)
            elif msg_type == "agent_message" and role == AGENT:
                await _handle_agent_message(ws, conversation_id, subject, data)
            else:
                await ws.send_json(_event("error", detail=f"unsupported frame: {msg_type!r}"))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WS conversation=%s crashed", conversation_id)
        try:
            await ws.close(code=CLOSE_SERVER_ERROR, reason="internal error")
        except Exception:
            logger.debug("WS conversation=%s: close after crash failed", conversation_id)
    finally:
        conversation_hub.leave(conversation_id, ws)


async def _handle_passenger_turn(ws: WebSocket, conversation_id: uuid.UUID, data: dict) -> None:
    content = (data.get("content") or "").strip()
    if not content:
        await ws.send_json(_event("error", detail="empty message"))
        return
    if len(content) > MAX_CONTENT_LENGTH:
        await ws.send_json(_event("error", detail="message too long"))
        return

    async def on_token(token_text: str) -> None:
        await ws.send_json(_event("bot_token", content=token_text))

    await ws.send_json(_event("typing", sender="bot"))
    try:
        result = await run_turn(conversation_id, content, on_token=on_token)
    except PermissionError as exc:
        await ws.send_json(_event("bot_error", detail=str(exc)))
        return
    except Exception:
        logger.exception("Turn failed for conversation=%s", conversation_id)
        await ws.send_json(_event("bot_error", detail="I couldn't process that. Please try again."))
        return

    # Room events: agents watching see the passenger's (sentiment-tagged) turn
    # and any bot answer; the sender dedupes by message id.
    await conversation_hub.send(
        conversation_id, _event("passenger_message", message=result["passenger_message"])
    )

    if result.get("escalate"):
        await conversation_hub.send(
            conversation_id,
            _event("escalated", reason=result.get("escalation_reason") or ""),
        )
        escalation_id = await _latest_escalation_id(conversation_id)
        await dashboard_hub.broadcast(
            _event(
                "queue_update",
                action="created",
                escalation_id=escalation_id,
                conversation_id=str(conversation_id),
            )
        )
        # No bot answer on the escalation path; acknowledge the turn anyway.
        await ws.send_json(_event("bot_done", message=None))
        return

    if result.get("bot_message"):
        await conversation_hub.send(
            conversation_id, _event("bot_message", message=result["bot_message"])
        )
        await ws.send_json(_event("bot_done", message=result["bot_message"]))
    else:
        await ws.send_json(_event("bot_error", detail="no answer generated"))


async def _handle_agent_message(
    ws: WebSocket, conversation_id: uuid.UUID, agent_id: uuid.UUID, data: dict
) -> None:
    content = (data.get("content") or "").strip()
    if not content or len(content) > MAX_CONTENT_LENGTH:
        await ws.send_json(_event("error", detail="invalid message"))
        return

    # Re-check the assignment on every message (the claim can be released).
    async with SessionLocal() as db:
        escalation = (
            await db.execute(
                select(Escalation).where(
                    Escalation.conversation_id == conversation_id,
                    Escalation.agent_id == agent_id,
                    Escalation.status == EscalationStatus.ASSIGNED,
                )
            )
        ).scalar_one_or_none()
        if escalation is None:
            await ws.send_json(_event("error", detail="you no longer own this escalation"))
            await ws.close(code=CLOSE_FORBIDDEN, reason="claim released")
            return
        message = Message(
            conversation_id=conversation_id, sender=MessageSender.AGENT, content=content
        )
        db.add(message)
        await db.commit()
        await db.refresh(message)

    await conversation_hub.send(
        conversation_id, _event("agent_message", message=message_to_dict(message))
    )


async def _latest_escalation_id(conversation_id: uuid.UUID) -> str | None:
    async with SessionLocal() as db:
        escalation = (
            await db.execute(
                select(Escalation)
                .where(Escalation.conversation_id == conversation_id)
                .order_by(Escalation.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    return str(escalation.id) if escalation else None


# --------------------------------------------------------------------------
# Dashboard channel
# --------------------------------------------------------------------------


@router.websocket("/ws/dashboard")
async def dashboard_socket(ws: WebSocket, token: str = Query(...)):
    auth = _decode_ws_token(token)
    if auth is None or auth[0] != AGENT:
        await ws.close(code=CLOSE_UNAUTHORIZED, reason="agent token required")
        return

    agent = auth[1]
    async with SessionLocal() as db:
        exists = (
            await db.execute(select(Agent.id).where(Agent.id == agent))
        ).scalar_one_or_none()
    if exists is None:
        await ws.close(code=CLOSE_UNAUTHORIZED, reason="agent not found")
        return

    await ws.accept()
    dashboard_hub.add(ws)
    await _start_heartbeat(ws)
    logger.info("WS dashboard agent=%s connected", agent)

    try:
        while True:
            data = await ws.receive_json()
            if data.get("type") == "ping":
                await ws.send_json(_event("pong"))
            # dashboards are receive-only besides heartbeats
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WS dashboard agent=%s crashed", agent)
    finally:
        dashboard_hub.remove(ws)
