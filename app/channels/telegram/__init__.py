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

The implementation lives in submodules (``helpers``, ``api``, ``mapping``,
``channel``); this package re-exports the public surface so existing imports
keep working unchanged.
"""

from .api import TelegramAPI, TelegramError
from .channel import TelegramChannel, get_telegram_channel
from .channel import _HubSink as _HubSink
from .helpers import (
    AGENT_PREFIX,
    CHUNK_LIMIT,
    CLOSED_NOTICE,
    ERROR_NOTICE,
    ESCALATION_NOTICE,
    HELP,
    TELEGRAM_MAX_CHARS,
    WELCOME,
    Incoming,
    chunk_text,
    parse_incoming,
    redact_secrets,
    to_telegram_html,
)
from .mapping import _get_or_create_conversation as _get_or_create_conversation
from .mapping import _get_or_create_passenger as _get_or_create_passenger

__all__ = [
    "AGENT_PREFIX",
    "CHUNK_LIMIT",
    "CLOSED_NOTICE",
    "ERROR_NOTICE",
    "ESCALATION_NOTICE",
    "HELP",
    "TELEGRAM_MAX_CHARS",
    "WELCOME",
    "Incoming",
    "TelegramAPI",
    "TelegramChannel",
    "TelegramError",
    "chunk_text",
    "get_telegram_channel",
    "parse_incoming",
    "redact_secrets",
    "to_telegram_html",
]
