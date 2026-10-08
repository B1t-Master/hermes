"""One-off: copy the Neon pgvector corpus into the local SqliteStore.

Used to seed the fast-local retrieval backend for demo environments where
the Neon endpoint has very high latency (e.g. sandboxes). It reuses the
existing chunks + embeddings exactly, so both stores are byte-identical and
no re-scraping / re-embedding happens.

Run:  .venv/bin/python scripts/seed_sqlite_store.py
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.db import SessionLocal, engine  # noqa: E402
from app.models import KnowledgeDocument, KnowledgeFragment  # noqa: E402
from ingest.store import SqliteStore  # noqa: E402

LOCAL_PATH = ".hermes_store.sqlite"


async def main() -> int:
    # Group fragments by source, preserving chunk order. NULL source_url in
    # legacy rows would violate the sqlite schema's NOT NULL, so tag those
    # with a stable synthetic key.
    docs: dict[str, dict] = defaultdict(
        lambda: {"title": "Untitled", "topic": "", "chunks": [], "embeddings": []}
    )
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(KnowledgeFragment, KnowledgeDocument.title)
                .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeFragment.document_id)
                .order_by(KnowledgeDocument.source_url, KnowledgeFragment.chunk_index)
            )
        ).all()
        for frag, title in rows:
            key = frag.source_url or f"untitled::{frag.document_id}"
            entry = docs[key]
            entry["title"] = title
            if frag.topic and not entry["topic"]:
                entry["topic"] = frag.topic
            entry["chunks"].append(frag.content)
            vector = frag.embedding
            if hasattr(vector, "tolist"):  # pgvector hands back a numpy array
                vector = vector.tolist()
            entry["embeddings"].append(list(vector) if vector is not None else None)

    store = SqliteStore(path=LOCAL_PATH)
    try:
        await store.connect()
        fresh = changed = 0
        for source_key, entry in docs.items():
            status = await store.replace_source(
                source_key=source_key,
                title=entry["title"],
                topic=entry["topic"],
                doc_type="doc",
                content_hash=source_key,  # identity marker; retrieval is by embedding
                chunks=entry["chunks"],
                embeddings=entry["embeddings"],
                source_url=source_key,
            )
            fresh += status == "new"
            changed += status == "updated"
        chunk_count = sum(len(e["chunks"]) for e in docs.values())
        print(f"seeded {LOCAL_PATH}: {len(docs)} documents "
              f"({fresh} new, {changed} updated), {chunk_count} chunks")
    finally:
        await store.close()
        await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))