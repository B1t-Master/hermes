import hashlib
import sys

from ingest.chunker import chunk_text
from ingest.embedder import Embedder
from ingest.parser import parse_source
from ingest.sources import Source
from ingest.store import Store


class IngestReport:
    def __init__(self):
        self.new = 0
        self.updated = 0
        self.unchanged = 0
        self.failed: list[tuple[str, str]] = []

    def summary(self) -> str:
        return (
            f"{self.new} new, {self.updated} updated, {self.unchanged} unchanged, "
            f"{len(self.failed)} failed"
        )


def content_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def ingest_sources(
    sources: list[Source],
    store: Store,
    *,
    embedder: Embedder | None = None,
    force: bool = False,
    embed: bool = True,
    warn: bool = True,
) -> IngestReport:
    report = IngestReport()
    can_embed = embed and embedder is not None and embedder.available
    if embed and embedder is not None and not embedder.available and warn:
        print(
            "[warn] sentence-transformers not installed; storing chunks without vectors. "
            "Install torch + sentence-transformers for real embeddings.",
            file=sys.stderr,
        )

    for source in sources:
        try:
            raw = parse_source(source)
        except ImportError as exc:
            report.failed.append((source.key, str(exc)))
            print(f"[warn] {source.key}: missing dependency ({exc})", file=sys.stderr)
            continue
        except Exception as exc:
            report.failed.append((source.key, repr(exc)))
            print(f"[warn] {source.key}: parse/scrape failed ({exc})", file=sys.stderr)
            continue

        if not raw.strip():
            report.failed.append((source.key, "empty content"))
            print(f"[warn] {source.key}: empty content, skipped", file=sys.stderr)
            continue

        hash_value = content_hash(raw)

        if not force:
            existing = await store.get_hash(source.key)
            if existing == hash_value:
                report.unchanged += 1
                print(f"[skip] {source.key} (unchanged)")
                continue

        chunks = chunk_text(raw)
        if not chunks:
            report.failed.append((source.key, "no chunks produced"))
            print(f"[warn] {source.key}: chunking produced nothing", file=sys.stderr)
            continue

        embeddings = None
        if can_embed:
            embeddings = embedder.embed(chunks)
        elif embed:
            embeddings = None

        status = await store.replace_source(
            source_key=source.key,
            title=source.title,
            topic=source.topic,
            doc_type=source.doc_type,
            content_hash=hash_value,
            chunks=chunks,
            embeddings=embeddings,
            source_url=source.source_url,
        )
        if status == "new":
            report.new += 1
        else:
            report.updated += 1
        print(f"[{status}] {source.key} ({len(chunks)} chunks)")

    return report