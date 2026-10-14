"""Stale-data refresh: content-hash diff between stored and newly fetched sources.

Only sources whose raw content changed are re-chunked / re-embedded; unchanged
sources are left untouched. Local PDFs refresh by simply re-running this script
after a file update (there is no watcher or cron yet).

Example
-------
    python -m ingest.refresh --embed
"""

import argparse
import asyncio

from ingest.embedder import Embedder
from ingest.pipeline import ingest_sources
from ingest.sources import discover_all
from ingest.store import get_store


async def refresh(options: argparse.Namespace) -> int:
    sources = discover_all(options.source)
    if not sources:
        print("No sources found.")
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
        print(f"Refresh complete. {report.summary()}")
        return 1 if report.failed else 0
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh changed knowledge sources")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--source", action="append", help="Extra source URL or file path")
    raise SystemExit(asyncio.run(refresh(parser.parse_args())))


if __name__ == "__main__":
    main()
