"""Full ingestion pipeline: parse, chunk, embed (optional), store.

Examples
--------
Ingest into local Postgres/pgvector (creates schema automatically):
    python -m ingest.run --embed

Force re-ingest of everything (ignore content hashes):
    python -m ingest.run --force
"""

import argparse
import asyncio

from ingest.embedder import Embedder
from ingest.pipeline import ingest_sources
from ingest.sources import discover_all
from ingest.store import get_store


async def run(options: argparse.Namespace) -> int:
    sources = discover_all(options.source)
    if not sources:
        print("No sources found. Drop PDFs/DOCX into data_sources/ or pass --source <url|path>.")
        return 1

    store = get_store()
    embedder = Embedder() if options.embed else None

    try:
        await store.ensure_schema()
        if options.reset:
            await store.reset()
        report = await ingest_sources(
            sources, store, embedder=embedder, force=options.force, embed=options.embed
        )
        print(f"Done. {report.summary()}")
        return 1 if report.failed else 0
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Compute embeddings (requires torch + sentence-transformers installed)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest every source even when its content hash is unchanged",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all knowledge documents/fragments before ingesting",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Extra source URL or file path (repeatable)",
    )
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
