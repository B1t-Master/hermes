"""Telegram channel adapter: polling loop, mapping and outbound delivery."""

from __future__ import annotations

import asyncio
import logging
import random
import traceback
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
    redact_secrets,
    to_telegram_html,
)
from .mapping import _get_or_create_conversation, _get_or_create_passenger

logger = logging.getLogger(__name__)

_BASE_RETRY_DELAY = 1.5
_MAX_RETRY_DELAY = 30.0
_WATCHDOG_AFTER = 120.0  # no successful getUpdates for 2 minutes => recycle the client
_MAX_POLL_TIMEOUT = 25  # longer long-polls get cut by some egress proxies


def _log_api_failure(action: str, exc: BaseException, delay: float | None = None) -> None:
    """Log a transient Bot API failure without the token-bearing request URL."""
    if isinstance(exc, httpx.HTTPStatusError):
        detail = f"HTTP {exc.response.status_code}"
    else:
        detail = type(exc).__name__
    message = redact_secrets(str(exc))
    if message:
        detail = f"{detail}: {message}"
    suffix = f"; retrying in {delay:.1f}s" if delay is not None else ""
    logger.warning("Telegram %s failed (%s)%s", action, detail, suffix)


def _log_exception(message: str, *args: object) -> None:
    """Log an unexpected error with a redacted traceback."""
    logger.error(
        "%s\n%s",
        redact_secrets(message % args if args else message),
        redact_secrets(traceback.format_exc()),
    )


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
        self._last_success = 0.0

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        self._closed = False
        try:
            await self._api.delete_webhook()
        except Exception as exc:
            _log_api_failure("deleteWebhook", exc)
            logger.warning("Telegram polling may conflict with a stale webhook")
        try:
            me = await self._api.get_me()
            logger.info("Telegram channel ready as @%s", (me or {}).get("username"))
        except Exception as exc:
            _log_api_failure("getMe", exc)
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
                _log_exception("Telegram poll loop raised on shutdown")
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
            _log_exception("Telegram: failed to load open conversations for rejoin")
            return
        for conversation_id, chat_id in rows:
            if not chat_id:
                continue
            self._join_room(conversation_id, int(chat_id))
        logger.info("Telegram rejoined %d open conversation(s)", len(rows))

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    def _retry_delay(self, failures: int) -> float:
        """Capped exponential backoff with 0-25% jitter."""
        base = min(_MAX_RETRY_DELAY, _BASE_RETRY_DELAY * 2 ** (failures - 1))
        return min(_MAX_RETRY_DELAY, base * (1 + random.random() * 0.25))

    async def _poll_loop(self) -> None:
        logger.info("Poll loop starting")
        offset = 0
        failures = 0
        self._last_success = self._now()
        while not self._closed:
            # A hung long-poll can leave the socket open with no data forever.
            # If we have not completed a single poll recently, recycle the
            # HTTP client so the next attempt uses a fresh connection.
            stalled_for = self._now() - self._last_success
            if stalled_for >= _WATCHDOG_AFTER:
                logger.warning(
                    "Telegram polling stalled for %.0fs; resetting HTTP client",
                    stalled_for,
                )
                await self._api.reset()
                self._last_success = self._now()

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
                failures += 1
                delay = self._retry_delay(failures)
                _log_api_failure("getUpdates", exc, delay)
                await asyncio.sleep(delay)
                continue
            except Exception as exc:
                failures += 1
                delay = self._retry_delay(failures)
                _log_api_failure("getUpdates", exc, delay)
                await asyncio.sleep(delay)
                continue

            failures = 0
            self._last_success = self._now()
            for update in updates or []:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = max(offset, update_id + 1)
                try:
                    await self._handle_update(update)
                except Exception:
                    _log_exception("Telegram update failed: %r", update)

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
            _log_exception("Telegram turn failed for conversation=%s", conversation_id)
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
        except Exception as exc:
            logger.debug("Telegram sendChatAction failed: %s", redact_secrets(str(exc)))

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
        poll_timeout = min(settings.telegram_poll_timeout, _MAX_POLL_TIMEOUT)
        if poll_timeout < settings.telegram_poll_timeout:
            logger.warning(
                "TELEGRAM_POLL_TIMEOUT=%s exceeds %ss; clamping to avoid proxy cutoffs",
                settings.telegram_poll_timeout,
                _MAX_POLL_TIMEOUT,
            )
        api = TelegramAPI(
            settings.telegram_bot_token,
            base_url=settings.telegram_api_base,
            timeout=poll_timeout + 15.0,
        )
        _channel = TelegramChannel(api, poll_timeout=poll_timeout)
    return _channel
