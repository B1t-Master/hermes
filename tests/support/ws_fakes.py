"""Shared WebSocket test helpers."""

from tests.support.fakes import FakeDB, FakeResult


class FakeSessionMaker:
    """Async context-manager stand-in for ``app.db.SessionLocal``."""

    def __init__(self, db):
        self.db = db
        self._session = None

    def __call__(self):
        return self

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, *exc_info):
        return False
