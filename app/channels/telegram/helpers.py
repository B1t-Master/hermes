"""Pure helpers for the Telegram adapter (unit-tested without network/DB)."""

from __future__ import annotations

import re
from dataclasses import dataclass

TELEGRAM_MAX_CHARS = 4096
# Leave headroom under the hard 4096 limit for the HTML we inject (<b></b>).
CHUNK_LIMIT = 3900

# httpx puts the full request URL in HTTPStatusError messages and tracebacks.
# Telegram URLs contain the bot token, so scrub it before anything is logged.
_BOT_TOKEN_RE = re.compile(r"/bot\d+:[A-Za-z0-9_-]{20,}")


def redact_secrets(text: str) -> str:
    """Replace any Telegram bot token in ``text`` with a safe placeholder."""
    return _BOT_TOKEN_RE.sub("/bot<redacted>", text)

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
