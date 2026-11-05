"""Telegram channel adapter: polling loop, mapping and outbound delivery."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable

import httpx
from sqlalchemy import select

from app.agents.hub import conversation_hub
from app.agents.notify import notify_turn
from app.agents.turns import run_turn
from app.config import settings
from app.db import SessionLocal
from app.models import Conversation, ConversationStatus, Passenger

from .api import TelegramAPI, TelegramError
from .helpers import (
    AGENT_PREFIX,
    CLOSED_NOTICE,
    ERROR_NOTICE,
    ESCALATION_NOTICE,
    HELP,
    WELCOME,
    chunk_text,
    parse_incoming,
    to_telegram_html,
)
from .mapping import _get_or_create_conversation, _get_or_create_passenger

logger = logging.getLogger(__name__)


class _HubSink:
    """Duck-typed conversation-room member that forwards agent replies.

    The hub only needs an object with an async ``send_json``; this filters
    to ``agent_message`` (human replies) and drops the events the adapter
    emits itself (bot/passenger messages), which would otherwise be sent to
    Telegram twice.
    """

    def __init__(self, on_agent_message: Callable[[str], Awaitable[None]]) -> None:
        self._on_agent_message = on_agent_message

    async def send_json(self, event: dict) -> None:
        if event.get("type") != "agent_message":
            return
        content = (event.get("message") or {}).get("content")
        if content:
            await self._on_agent_message(content)


class TelegramChannel:
    def __init__(self, api: TelegramAPI, poll_timeout: int = 25) -> None:
        self._api = api
        self._poll_timeout = poll_timeout
        self._task: asyncio.Task | None = None
        self._sinks: dict[uuid.UUID, _HubSink] = {}
        self._closed = False

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        self._closed = False
        try:
            await self._api.delete_webhook()
        except Exception:
            logger.warning("Telegram: deleteWebhook failed; polling may conflict", exc_info=True)
        try:
            me = await self._api.get_me()
            logger.info("Telegram channel ready as @%s", (me or {}).get("username"))
        except Exception:
            logger.warning("Telegram: getMe failed", exc_info=True)
        await self._rejoin_rooms()
        self._task = asyncio.create_task(self._poll_loop(), name="telegram-poll")
        logger.info("Telegram channel started (long polling)")

    async def stop(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Telegram poll loop raised on shutdown", exc_info=True)
            self._task = None
        for conversation_id, sink in list(self._sinks.items()):
            conversation_hub.leave(conversation_id, sink)
        self._sinks.clear()
        await self._api.close()

    # -- polling ---------------------------------------------------------

    async def _rejoin_rooms(self) -> None:
        """Re-attach sinks to open Telegram conversations after a restart.

        The conversation hub is in-process and its room members vanish when
        the backend restarts. Without this, an agent reply is broadcast to a
        room the adapter is not part of and is silently dropped until the
        passenger happens to send another message.
        """
        try:
            async with SessionLocal() as db:
                rows = (
                    await db.execute(
                        select(Conversation.id, Passenger.telegram_chat_id)
                        .join(Passenger, Passenger.id == Conversation.passenger_id)
                        .where(
                            Passenger.telegram_chat_id.isnot(None),
                            Conversation.status != ConversationStatus.RESOLVED,
                        )
                    )
                ).all()
        except Exception:
            logger.exception("Telegram: failed to load open conversations for rejoin")
            return
        for conversation_id, chat_id in rows:
            if not chat_id:
                continue
            self._join_room(conversation_id, int(chat_id))
        logger.info("Telegram rejoined %d open conversation(s)", len(rows))

    async def _poll_loop(self) -> None:
        logger.info("Poll loop starting")
        offset = 0
        while not self._closed:
            logger.debug("Polling for updates (offset=%d)", offset)
            try:
                updates = await self._api.get_updates(offset, timeout=self._poll_timeout)
            except asyncio.CancelledError:
                raise
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 409:
                    # Another getUpdates connection is still active on
                    # Telegram's side (usually a stale long-poll from a
                    # previous instance). Wait for it to expire before
                    # retrying, rather than hammering the API.
                    logger.warning("Telegram 409 Conflict; waiting 30s for stale connection to expire")
                    await asyncio.sleep(30)
                    continue
                logger.exception("Telegram getUpdates failed; retrying in 3s")
                await asyncio.sleep(3)
                continue
            except Exception:
                logger.exception("Telegram getUpdates failed; retrying in 3s")
                await asyncio.sleep(3)
                continue
            for update in updates or []:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = max(offset, update_id + 1)
                try:
                    await self._handle_update(update)
                except Exception:
                    logger.exception("Telegram update failed: %r", update)

    async def _handle_update(self, update: dict) -> None:
        incoming = parse_incoming(update)
        if incoming is None:
            return
        text = incoming.text
        if text.startswith("/"):
            command = text.split()[0].split("@")[0].lower()
            if command in ("/start", "/help"):
                reply = WELCOME if command == "/start" else HELP
                await self._send_text(incoming.chat_id, reply)
                return
            # Unknown command: fall through and let the bot answer it.
        await self._send_typing(incoming.chat_id)
        await self._handle_text(incoming.chat_id, text, incoming.name)

    async def _handle_text(self, chat_id: int, text: str, name: str) -> None:
        passenger_id = await _get_or_create_passenger(chat_id, name)
        conversation_id = await _get_or_create_conversation(passenger_id)
        self._join_room(conversation_id, chat_id)

        try:
            result = await run_turn(conversation_id, text)
        except PermissionError:
            await self._send_text(chat_id, CLOSED_NOTICE)
            return
        except Exception:
            logger.exception("Telegram turn failed for conversation=%s", conversation_id)
            await self._send_text(chat_id, ERROR_NOTICE)
            return

        await notify_turn(conversation_id, result)

        if result.get("escalate"):
            # Only announce a genuinely new escalation; on an already-escalated
            # conversation the human agent owns the reply.
            if result.get("escalation_created"):
                await self._send_text(chat_id, ESCALATION_NOTICE)
            else:
                # Already escalated: acknowledge the message so the passenger
                # is not left staring at a typing indicator with no reply.
                await self._send_text(
                    chat_id,
                    "Your conversation has been escalated to a human agent — "
                    "they'll reply here as soon as they can. You can keep "
                    "typing; your messages go straight to them.",
                )
            return

        bot = result.get("bot_message")
        if bot and bot.get("content"):
            await self._send_text(chat_id, bot["content"])
        else:
            await self._send_text(chat_id, "Sorry, I couldn't put an answer together. Could you rephrase?")

    def _join_room(self, conversation_id: uuid.UUID, chat_id: int) -> None:
        if conversation_id in self._sinks:
            return
        sink = _HubSink(lambda content: self._send_text(chat_id, AGENT_PREFIX + content))
        self._sinks[conversation_id] = sink
        conversation_hub.join(conversation_id, sink)  # type: ignore[arg-type]

    # -- outbound --------------------------------------------------------

    async def _send_typing(self, chat_id: int) -> None:
        try:
            await self._api.send_chat_action(chat_id, "typing")
        except Exception:
            logger.debug("Telegram sendChatAction failed", exc_info=True)

    async def _send_text(self, chat_id: int, text: str) -> None:
        for chunk in chunk_text(text):
            try:
                await self._api.send_message(chat_id, to_telegram_html(chunk), parse_mode="HTML")
            except (TelegramError, httpx.HTTPError):
                # Unbalanced markup (or a transient API error): retry once as
                # plain text so the passenger still gets the answer.
                await self._api.send_message(chat_id, chunk)


# --------------------------------------------------------------------------
# Singleton
# --------------------------------------------------------------------------

_channel: TelegramChannel | None = None


def get_telegram_channel() -> TelegramChannel:
    global _channel
    if _channel is None:
        api = TelegramAPI(
            settings.telegram_bot_token,
            base_url=settings.telegram_api_base,
            timeout=settings.telegram_poll_timeout + 15.0,
        )
        _channel = TelegramChannel(api, poll_timeout=settings.telegram_poll_timeout)
    return _channel
