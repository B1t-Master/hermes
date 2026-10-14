import asyncio
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime

from app.config import settings

# In-process signal that the stored corpus changed. CachedStore compares its
# loaded version against this so a KB mutation is visible on the very next
# search without a network round trip per turn.
_corpus_version = 0


def bump_corpus_version() -> None:
    """Mark the corpus as changed; in-process caches reload on next search."""
    global _corpus_version
    _corpus_version += 1


def corpus_version() -> int:
    return _corpus_version


def keyword_terms(query: str) -> list[str]:
    """Distinct meaningful terms from a query, for the no-embedding fallback."""
    import re
    return sorted({t for t in re.findall(r"\w+", query.lower()) if len(t) > 2})


def keyword_score(content: str, terms: list[str]) -> float:
    """Fraction of query terms present in the content (0.0 if no terms)."""
    if not terms:
        return 0.0
    haystack = content.lower()
    hits = sum(1 for t in terms if t in haystack)
    return hits / len(terms)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class RetrievedChunk:
    """One search hit, shaped for the QA agent's context window."""

    content: str
    source_url: str | None
    title: str | None
    topic: str | None
    chunk_index: int
    score: float  # cosine similarity in [0, 1]; keyword mode: query-term overlap ratio

    @property
    def citation(self) -> str:
        """Short human-readable attribution for answer footnotes."""
        label = self.title or self.source_url or "unknown source"
        return f"[{label}]" if self.source_url is None else f"[{label}] ({self.source_url})"


@dataclass
class Fragment:
    """One stored chunk, decoupled from any backend, for in-process ranking."""

    content: str
    source_url: str | None
    title: str | None
    topic: str | None
    chunk_index: int
    embedding: list[float] | None


def rank_fragments(
    fragments: list[Fragment],
    query_embedding: list[float] | None,
    query_text: str,
    k: int = 5,
) -> list[RetrievedChunk]:
    """Score `fragments` against the query and return the top-k.

    Vector mode when `query_embedding` is given; otherwise keyword-overlap
    fallback. Shared by the stores and the in-process cache so every path ranks
    identically.
    """
    terms = None if query_embedding is not None else keyword_terms(query_text)
    results: list[RetrievedChunk] = []
    for fragment in fragments:
        if query_embedding is not None:
            if not fragment.embedding:
                continue  # unembedded chunk can't be cosine-ranked
            score = cosine_similarity(query_embedding, fragment.embedding)
        else:
            score = keyword_score(fragment.content, terms or [])
            if score <= 0:
                continue
        results.append(
            RetrievedChunk(
                content=fragment.content,
                source_url=fragment.source_url,
                title=fragment.title,
                topic=fragment.topic,
                chunk_index=fragment.chunk_index,
                score=score,
            )
        )
    results.sort(key=lambda chunk: chunk.score, reverse=True)
    return results[:k]


class Store(ABC):
    @abstractmethod
    async def get_hash(self, source_key: str) -> str | None: ...

    @abstractmethod
    async def replace_source(
        self, *, source_key: str, title: str, topic: str, doc_type: str, content_hash: str,
        chunks: list[str], embeddings: list[list[float] | None] | None, source_url: str | None,
    ) -> str: ...

    @abstractmethod
    async def list_documents(self) -> list[dict]:
        """Every stored document as a dict with keys: id, source_url, title,
        doc_type, content_hash, last_fetched, created_at, fragment_count."""

    @abstractmethod
    async def delete_source(self, source_key: str) -> bool:
        """Delete a document and its fragments; False when it did not exist."""

    @abstractmethod
    async def dump_fragments(self) -> list[Fragment]:
        """Every stored chunk with its embedding, for an in-process index."""

    @abstractmethod
    async def ensure_schema(self) -> None: ...

    @abstractmethod
    async def reset(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float] | None,
        query_text: str,
        k: int = 5,
    ) -> list[RetrievedChunk]:
        """Top-k chunks by cosine similarity; keyword-overlap fallback when
        `query_embedding` is None (corpus or query not embedded)."""


class PgVectorStore(Store):
    """PostgreSQL + pgvector backend.

    Reuses the app ORM (app.models) and the async engine from app.db.
    `source_key` is stored in the knowledge_documents.source_url column
    (values like 'url::https://...' or 'file::C:/...').
    """

    def __init__(self):
        from app.db import SessionLocal, engine
        from app.models import KnowledgeDocument, KnowledgeFragment

        self._SessionLocal = SessionLocal
        self._engine = engine
        self._KnowledgeDocument = KnowledgeDocument
        self._KnowledgeFragment = KnowledgeFragment

    async def get_hash(self, source_key: str) -> str | None:
        from sqlalchemy import select

        async with self._SessionLocal() as session:
            result = await session.execute(
                select(self._KnowledgeDocument.content_hash).where(
                    self._KnowledgeDocument.source_url == source_key
                )
            )
            return result.scalar_one_or_none()

    async def ensure_schema(self) -> None:
        from app.db import init_models

        await init_models()

    async def reset(self) -> None:
        """Drop all knowledge content (documents + fragments only)."""
        from sqlalchemy import delete

        async with self._SessionLocal() as session:
            await session.execute(delete(self._KnowledgeDocument))
            await session.execute(delete(self._KnowledgeFragment))
            await session.commit()

    async def replace_source(self, *, source_key, title, topic, doc_type, content_hash,
                             chunks, embeddings, source_url) -> str:
        from sqlalchemy import select

        embeddings = embeddings or [None] * len(chunks)
        now = datetime.now(UTC)

        async with self._SessionLocal() as session:
            result = await session.execute(
                select(self._KnowledgeDocument).where(
                    self._KnowledgeDocument.source_url == source_key
                )
            )
            existing = result.scalar_one_or_none()
            if existing is not None:
                await session.delete(existing)
                await session.flush()
                status = "updated"
            else:
                status = "new"

            document = self._KnowledgeDocument(
                source_url=source_key,
                title=title,
                doc_type=doc_type,
                content_hash=content_hash,
                last_fetched=now,
            )
            session.add(document)
            await session.flush()

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
                session.add(
                    self._KnowledgeFragment(
                        document_id=document.id,
                        chunk_index=i,
                        content=chunk,
                        embedding=embedding,
                        source_url=source_url,
                        topic=topic,
                        last_fetched=now,
                    )
                )
            await session.commit()
            return status

    async def list_documents(self) -> list[dict]:
        from sqlalchemy import func, select

        frag_count = func.count(self._KnowledgeFragment.id).label("fragment_count")
        stmt = (
            select(self._KnowledgeDocument, frag_count)
            .outerjoin(
                self._KnowledgeFragment,
                self._KnowledgeFragment.document_id == self._KnowledgeDocument.id,
            )
            .group_by(self._KnowledgeDocument.id)
            .order_by(self._KnowledgeDocument.created_at.desc())
        )
        async with self._SessionLocal() as session:
            rows = (await session.execute(stmt)).all()
        return [
            {
                "id": str(doc.id),
                "source_url": doc.source_url,
                "title": doc.title,
                "doc_type": doc.doc_type,
                "content_hash": doc.content_hash,
                "last_fetched": doc.last_fetched,
                "created_at": doc.created_at,
                "fragment_count": int(fc or 0),
            }
            for doc, fc in rows
        ]

    async def delete_source(self, source_key: str) -> bool:
        from sqlalchemy import select

        async with self._SessionLocal() as session:
            doc = (
                await session.execute(
                    select(self._KnowledgeDocument).where(
                        self._KnowledgeDocument.source_url == source_key
                    )
                )
            ).scalar_one_or_none()
            if doc is None:
                return False
            await session.delete(doc)
            await session.commit()
        return True

    async def dump_fragments(self) -> list[Fragment]:
        from sqlalchemy import select

        frag = self._KnowledgeFragment
        doc = self._KnowledgeDocument
        stmt = select(
            frag.content,
            frag.source_url,
            frag.topic,
            frag.chunk_index,
            frag.embedding,
            doc.title,
        ).join(doc, doc.id == frag.document_id)
        async with self._SessionLocal() as session:
            rows = (await session.execute(stmt)).all()
        return [
            Fragment(
                content=content,
                source_url=source_url,
                title=title,
                topic=topic,
                chunk_index=chunk_index,
                embedding=[float(x) for x in embedding] if embedding is not None else None,
            )
            for content, source_url, topic, chunk_index, embedding, title in rows
        ]

    async def search(
        self, query_embedding: list[float] | None, query_text: str, k: int = 5
    ) -> list[RetrievedChunk]:
        from sqlalchemy import or_, select

        frag = self._KnowledgeFragment
        doc = self._KnowledgeDocument

        async with self._SessionLocal() as session:
            if query_embedding:
                distance = frag.embedding.cosine_distance(query_embedding)
                stmt = (
                    select(frag, doc.title, distance.label("distance"))
                    .join(doc, doc.id == frag.document_id)
                    .where(frag.embedding.is_not(None))
                    .order_by(distance)
                    .limit(k)
                )
                rows = (await session.execute(stmt)).all()
                return [
                    RetrievedChunk(
                        content=fragment.content,
                        source_url=fragment.source_url,
                        title=title,
                        topic=fragment.topic,
                        chunk_index=fragment.chunk_index,
                        score=1.0 - float(distance),
                    )
                    for fragment, title, distance in rows
                ]

            # Keyword fallback: corpus has no embeddings (or embedder missing)
            terms = keyword_terms(query_text)
            if not terms:
                return []
            conditions = [frag.content.ilike(f"%{term}%") for term in terms]
            stmt = (
                select(frag, doc.title)
                .join(doc, doc.id == frag.document_id)
                .where(or_(*conditions))
                .limit(200)
            )
            rows = (await session.execute(stmt)).all()
            scored = sorted(
                ((keyword_score(fragment.content, terms), fragment, title) for fragment, title in rows),
                key=lambda item: item[0],
                reverse=True,
            )
            return [
                RetrievedChunk(
                    content=fragment.content,
                    source_url=fragment.source_url,
                    title=title,
                    topic=fragment.topic,
                    chunk_index=fragment.chunk_index,
                    score=score,
                )
                for score, fragment, title in scored[:k]
                if score > 0
            ]

    async def close(self) -> None:
        await self._engine.dispose()


class CachedStore(Store):
    """Wraps PgVectorStore and serves ``search`` from an in-process index.

    The corpus is small and changes rarely, so loading every embedded chunk
    once and ranking in-process removes a network round trip from every turn.
    Staleness is detected via the process-wide corpus version (bumped by KB
    mutations) plus an optional TTL, so out-of-band edits self-heal.
    """

    def __init__(self, inner: PgVectorStore, ttl_seconds: float = 300.0):
        self._inner = inner
        self._ttl = ttl_seconds
        self._fragments: list[Fragment] | None = None
        self._loaded_at = 0.0
        self._loaded_version = -1
        self._lock = asyncio.Lock()

    # --- pass-through mutations: delegate, then invalidate the local index ---

    async def get_hash(self, source_key: str) -> str | None:
        return await self._inner.get_hash(source_key)

    async def ensure_schema(self) -> None:
        await self._inner.ensure_schema()

    async def list_documents(self) -> list[dict]:
        return await self._inner.list_documents()

    async def replace_source(self, **kwargs) -> str:
        status = await self._inner.replace_source(**kwargs)
        self.invalidate()
        return status

    async def delete_source(self, source_key: str) -> bool:
        found = await self._inner.delete_source(source_key)
        if found:
            self.invalidate()
        return found

    async def reset(self) -> None:
        await self._inner.reset()
        self.invalidate()

    async def close(self) -> None:
        await self._inner.close()

    # --- cached reads ---

    def invalidate(self) -> None:
        """Drop the index; the next search reloads it."""
        self._fragments = None
        self._loaded_version = -1

    def _stale(self) -> bool:
        if self._fragments is None:
            return True
        if corpus_version() != self._loaded_version:
            return True
        return self._ttl > 0 and (time.monotonic() - self._loaded_at) > self._ttl

    async def _ensure_fresh(self) -> list[Fragment]:
        if self._stale():
            async with self._lock:
                if self._stale():
                    self._fragments = await self._inner.dump_fragments()
                    self._loaded_at = time.monotonic()
                    self._loaded_version = corpus_version()
        return self._fragments or []

    async def dump_fragments(self) -> list[Fragment]:
        return list(await self._ensure_fresh())

    async def search(
        self, query_embedding: list[float] | None, query_text: str, k: int = 5
    ) -> list[RetrievedChunk]:
        fragments = await self._ensure_fresh()
        return rank_fragments(fragments, query_embedding, query_text, k)


def get_store() -> CachedStore:
    """Factory for the serving store: CachedStore over PgVectorStore."""
    inner = PgVectorStore()
    return CachedStore(inner, ttl_seconds=settings.kb_cache_ttl_seconds)
