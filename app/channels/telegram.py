"""Telegram channel adapter.

Bridges Telegram chats onto the same ``run_turn`` pipeline the web chat
uses, so a passenger gets identical answers (and escalations) on either
channel.

Transport is **long polling** (``getUpdates``), not webhooks: it needs no
public URL/tunnel and works on a laptop. A production deployment behind a
load balancer could swap in a webhook handler; the mapping and turn logic
here would be unchanged.

Mapping
-------
One Telegram chat -> one ``Passenger`` (``telegram_chat_id``) -> their most
recent non-resolved ``Conversation``. That keeps multi-turn history and the
escalation lifecycle intact across restarts.

Human agent replies
-------------------
When the bot escalates, a human answers from the agent dashboard over the
conversation WebSocket. This adapter is not a WS client, so it joins the
conversation room with a duck-typed sink (an object with ``send_json``)
and forwards ``agent_message`` events back to Telegram. Room events the
adapter already produces itself (bot/passenger messages) are ignored, so
nothing is delivered twice.
"""

import asyncio
import logging
import random
import re
import traceback
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx
from sqlalchemy import select

from app.agents.hub import conversation_hub
from app.agents.notify import notify_turn
from app.agents.turns import run_turn
from app.config import settings
from app.db import SessionLocal
from app.models import Conversation, ConversationStatus, Passenger

logger = logging.getLogger(__name__)

TELEGRAM_MAX_CHARS = 4096
# Leave headroom under the hard 4096 limit for the HTML we inject (<b></b>).
CHUNK_LIMIT = 3900

WELCOME = (
    "Hi! I'm the Kenya Airways assistant. Ask me about baggage, rebooking, "
    "refunds, check-in and more — or send /help. If I can't help, I can pass "
    "you to a human agent."
)
HELP = (
    "Just send your question as a normal message. Type /start to begin again."
)
ESCALATION_NOTICE = (
    "I've passed this to a human agent — they'll reply here as soon as they "
    "can. You can keep typing; your messages go straight to them."
)
CLOSED_NOTICE = "This conversation is closed. Send /start to begin a new one."
ERROR_NOTICE = "Sorry, something went wrong on my side. Please try again."
AGENT_PREFIX = "👤 Agent: "

_BASE_RETRY_DELAY = 1.5
_MAX_RETRY_DELAY = 30.0
_WATCHDOG_AFTER = 120.0  # no successful getUpdates for 2 minutes => recycle the client
_MAX_POLL_TIMEOUT = 25  # longer long-polls get cut by some egress proxies

# httpx puts the full request URL in HTTPStatusError messages and tracebacks.
# Telegram URLs contain the bot token, so scrub it before anything is logged.
_BOT_TOKEN_RE = re.compile(r"/bot\d+:[A-Za-z0-9_-]{20,}")


def redact_secrets(text: str) -> str:
    """Replace any Telegram bot token in ``text`` with a safe placeholder."""
    return _BOT_TOKEN_RE.sub("/bot<redacted>", text)


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


# --------------------------------------------------------------------------
# Pure helpers (unit-tested without network/DB)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Incoming:
    """A usable text message pulled out of a Telegram update."""

    chat_id: int
    text: str
    name: str


def parse_incoming(update: dict) -> Incoming | None:
    """Extract (chat_id, text, name) from an update, or None.

    Only plain text messages qualify; stickers, photos, edits, edited
    messages and channel posts all return None.
    """
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    text = message.get("text")
    if chat_id is None or not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    sender = message.get("from") or {}
    name = sender.get("first_name") or chat.get("first_name") or chat.get("title") or ""
    return Incoming(chat_id=int(chat_id), text=text, name=str(name))


def chunk_text(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` chars.

    Prefers a newline boundary, then a space, then a hard cut, so long
    answers (which Telegram rejects past 4096 chars) survive intact.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


_HTML_ESCAPE = {ord("&"): "&amp;", ord("<"): "&lt;", ord(">"): "&gt;"}
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def to_telegram_html(text: str) -> str:
    """Escape HTML, then re-enable Markdown ``**bold**`` as ``<b>``.

    The LLM emits Markdown; Telegram's HTML mode is the least error-prone
    way to render a subset of it. Anything unbalanced is caught by the
    caller, which retries as plain text.
    """
    return _BOLD_RE.sub(r"<b>\1</b>", text.translate(_HTML_ESCAPE))


# --------------------------------------------------------------------------
# Telegram Bot API client
# --------------------------------------------------------------------------


class TelegramError(RuntimeError):
    """The Bot API answered ``ok: false``."""


class TelegramAPI:
    """Thin async wrapper over the subset of the Bot API we use."""

    def __init__(
        self,
        token: str,
        base_url: str = "https://api.telegram.org",
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = f"{base_url.rstrip('/')}/bot{token}"
        self._timeout = timeout
        self._client = client or self._new_client()
        self._owns_client = client is None

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._timeout,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        )

    async def _call(self, method: str, payload: dict) -> object:
        response = await self._client.post(f"{self._base}/{method}", json=payload)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise TelegramError(data.get("description") or f"{method} failed")
        return data.get("result")

    async def get_me(self) -> dict:
        return await self._call("getMe", {})  # type: ignore[return-value]

    async def get_updates(self, offset: int, timeout: int) -> list[dict]:
        return await self._call(  # type: ignore[return-value]
            "getUpdates",
            {"offset": offset, "timeout": timeout, "allowed_updates": ["message"]},
        )

    async def send_message(self, chat_id: int, text: str, parse_mode: str | None = None) -> dict:
        payload: dict = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        return await self._call("sendMessage", payload)  # type: ignore[return-value]

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> object:
        return await self._call("sendChatAction", {"chat_id": chat_id, "action": action})

    async def delete_webhook(self) -> object:
        return await self._call("deleteWebhook", {"drop_pending_updates": False})

    async def reset(self) -> None:
        """Drop the current HTTP connection and start a fresh client.

        A long-poll socket can hang without ever returning headers (NAT/
        proxy timeout). Recycling the client forces new TCP/TLS connections
        the next time we poll.
        """
        if self._owns_client:
            await self._client.aclose()
            self._client = self._new_client()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


# --------------------------------------------------------------------------
# DB mapping: Telegram chat -> passenger -> conversation
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Channel
# --------------------------------------------------------------------------


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
