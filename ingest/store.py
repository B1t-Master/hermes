import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime

from app.config import settings

LOCAL_SQLITE_PATH = ".hermes_store.sqlite"


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


def keyword_terms(query: str) -> list[str]:
    """Distinct meaningful terms from a query, for the no-embedding fallback."""
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


class Store(ABC):
    @abstractmethod
    async def get_hash(self, source_key: str) -> str | None: ...

    @abstractmethod
    async def replace_source(
        self, *, source_key: str, title: str, topic: str, doc_type: str, content_hash: str,
        chunks: list[str], embeddings: list[list[float] | None] | None, source_url: str | None,
    ) -> str: ...

    async def ensure_schema(self) -> None:  # noqa: B027 - intentional default
        """Create/verify tables & extension. No-op by default (SqliteStore
        builds its schema in connect())."""

    async def reset(self) -> None:  # noqa: B027 - intentional default
        """Drop stored knowledge content (not part of default flow; --reset flag)."""

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
    """PostgreSQL + pgvector backend (production path).

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
                # Flush the DELETE before inserting: the new row carries the
                # same content_hash and that column is UNIQUE.
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


class SqliteStore(Store):
    """Local SQLite stand-in for development when the Neon endpoint is
    unreachable (e.g. networks that block outbound :5432). Never used in
    production; embeddings are stored as JSON so vector search is computed
    in-process during Phase 4 retrievals."""

    def __init__(self, path: str = LOCAL_SQLITE_PATH):
        self._path = path
        self._db = None

    async def connect(self) -> None:
        import aiosqlite

        self._db = await aiosqlite.connect(self._path)
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                source_key TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                topic TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                last_fetched TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS fragments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_key TEXT NOT NULL REFERENCES documents(source_key) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                source_url TEXT,
                topic TEXT,
                embedding_json TEXT
            )
            """
        )
        await self._db.commit()

    async def get_hash(self, source_key: str) -> str | None:
        cursor = await self._db.execute(
            "SELECT content_hash FROM documents WHERE source_key = ?", (source_key,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row[0] if row else None

    async def replace_source(self, *, source_key, title, topic, doc_type, content_hash,
                             chunks, embeddings, source_url) -> str:
        embeddings = embeddings or [None] * len(chunks)
        cursor = await self._db.execute(
            "SELECT 1 FROM documents WHERE source_key = ?", (source_key,)
        )
        existing = await cursor.fetchone()
        await cursor.close()
        status = "updated" if existing else "new"
        if existing:
            await self._db.execute("DELETE FROM documents WHERE source_key = ?", (source_key,))

        await self._db.execute(
            """INSERT INTO documents (source_key, title, topic, doc_type, content_hash, last_fetched)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (source_key, title, topic, doc_type, content_hash, datetime.now(UTC).isoformat()),
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            await self._db.execute(
                """INSERT INTO fragments
                   (source_key, chunk_index, content, source_url, topic, embedding_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    source_key,
                    i,
                    chunk,
                    source_url,
                    topic,
                    json.dumps(embedding) if embedding is not None else None,
                ),
            )
        await self._db.commit()
        return status

    async def reset(self) -> None:
        await self._db.execute("DELETE FROM documents")
        await self._db.commit()

    async def search(
        self, query_embedding: list[float] | None, query_text: str, k: int = 5
    ) -> list[RetrievedChunk]:
        cursor = await self._db.execute(
            """SELECT f.content, f.source_url, f.topic, f.chunk_index, f.embedding_json, d.title
               FROM fragments f JOIN documents d ON d.source_key = f.source_key"""
        )
        rows = await cursor.fetchall()
        await cursor.close()

        terms = None if query_embedding is not None else keyword_terms(query_text)
        results: list[RetrievedChunk] = []
        for content, source_url, topic, chunk_index, embedding_json, title in rows:
            if query_embedding is not None:
                if not embedding_json:
                    continue  # unembedded chunk can't be cosine-ranked
                try:
                    embedding = json.loads(embedding_json)
                except json.JSONDecodeError:
                    continue
                score = cosine_similarity(query_embedding, embedding)
            else:
                score = keyword_score(content, terms or [])
                if score <= 0:
                    continue
            results.append(
                RetrievedChunk(
                    content=content,
                    source_url=source_url,
                    title=title,
                    topic=topic,
                    chunk_index=chunk_index,
                    score=score,
                )
            )
        results.sort(key=lambda chunk: chunk.score, reverse=True)
        return results[:k]

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()


async def resolve_store(store_type: str | None = None) -> Store:
    """'auto' (default): PgVectorStore when DATABASE_URL is set, else SqliteStore.
    'pg' / 'sqlite' force a backend."""
    import os

    if store_type is None:
        store_type = os.environ.get("INGEST_STORE", "auto")

    has_database_url = settings.database_url and not settings.database_url.startswith(
        "postgresql+asyncpg://localhost"
    )
    if store_type == "pg" or (store_type == "auto" and has_database_url):
        return PgVectorStore()

    if store_type in {"sqlite", "auto"}:
        store = SqliteStore()
        await store.connect()
        return store

    raise ValueError(f"Unknown store type: {store_type}")