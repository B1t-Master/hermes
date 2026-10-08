"""Retrieval: query -> embedding -> store search, with keyword fallback.

Single entry point for the QA agent. Handles the messy reality that the
embedding stack may be missing or fail at runtime; retrieval degrades to
keyword-overlap search instead of crashing the answer path.
"""

import asyncio
import logging

from ingest.embedder import Embedder
from ingest.store import RetrievedChunk, resolve_store

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, k: int = 5):
        self.k = k
        self._store = None
        self._embedder: Embedder | None = None

    async def _get_store(self):
        if self._store is None:
            self._store = await resolve_store("auto")
        return self._store

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Top-k chunks for `query`; [] only when the corpus itself is empty."""
        embedding = await self._embed(query)
        if embedding is None:
            logger.info("No query embedding available; using keyword search")
        store = await self._get_store()
        return await store.search(embedding, query, k=self.k)

    async def _embed(self, query: str) -> list[float] | None:
        """Query embedding, or None to trigger the store's keyword fallback."""
        embedder = self._get_embedder()
        if not embedder.available:
            return None
        try:
            vectors = await asyncio.to_thread(embedder.embed, [query])
        except Exception:
            logger.exception("Query embedding failed; using keyword search")
            return None
        return vectors[0] if vectors else None

    async def close(self) -> None:
        if self._store is not None:
            await self._store.close()
            self._store = None


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Shared instance so the embedding model loads at most once."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
