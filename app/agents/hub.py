"""In-memory WebSocket registries.

Two hubs:
  - ConversationHub: deliver events to everyone watching a conversation
    (the passenger, plus a human agent once escalation happens).
  - DashboardHub: broadcast queue events to every connected agent dashboard.

Prototype-grade: in-process only. A multi-process deployment would swap these
for Redis pub/sub; the interfaces stay the same.
"""

import logging
import uuid
from collections import defaultdict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class _SocketSet:
    def __init__(self) -> None:
        self._sockets: set[WebSocket] = set()

    def add(self, ws: WebSocket) -> None:
        self._sockets.add(ws)

    def remove(self, ws: WebSocket) -> None:
        self._sockets.discard(ws)

    async def broadcast(self, event: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._sockets):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
                self._sockets.discard(ws)

    def __len__(self) -> int:
        return len(self._sockets)


class ConversationHub:
    def __init__(self) -> None:
        self._rooms: dict[uuid.UUID, _SocketSet] = defaultdict(_SocketSet)

    def join(self, conversation_id: uuid.UUID, ws: WebSocket) -> None:
        self._rooms[conversation_id].add(ws)

    def leave(self, conversation_id: uuid.UUID, ws: WebSocket) -> None:
        room = self._rooms.get(conversation_id)
        if room is None:
            return
        room.remove(ws)
        if len(room) == 0:
            del self._rooms[conversation_id]

    async def send(self, conversation_id: uuid.UUID, event: dict) -> None:
        room = self._rooms.get(conversation_id)
        if room is not None:
            await room.broadcast(event)


class DashboardHub(_SocketSet):
    async def broadcast(self, event: dict) -> None:  # type: ignore[override]
        await super().broadcast(event)


conversation_hub = ConversationHub()
dashboard_hub = DashboardHub()
