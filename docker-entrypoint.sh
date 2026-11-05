#!/bin/sh
set -eu

SEED_DIR="${SEED_SOURCES_DIR:-/app/seed_data_sources}"
DATA_DIR="${KNOWLEDGE_SOURCES_DIR:-/app/data_sources}"

# Bundled documents live in the image (read-only seed); uploads live in the
# writable named volume. Refresh the bundled files on every start so a rebuild
# actually updates the knowledge sources, while leaving uploaded files alone.
if [ -d "$SEED_DIR" ]; then
    mkdir -p "$DATA_DIR"
    cp -a "$SEED_DIR"/. "$DATA_DIR"/ || \
        echo "[entrypoint] warning: could not sync bundled sources into $DATA_DIR" >&2
fi

# Hand PID 1 to the real process so signals (SIGTERM) reach uvicorn/ingest.
exec "$@"
