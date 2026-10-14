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
import re
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
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

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

    async def _poll_loop(self) -> None:
        offset = 0
        while not self._closed:
            try:
                updates = await self._api.get_updates(offset, timeout=self._poll_timeout)
            except asyncio.CancelledError:
                raise
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
