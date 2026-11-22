"""Thin async wrapper over the subset of the Telegram Bot API we use."""

from __future__ import annotations

import httpx


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
