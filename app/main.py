import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import agents, auth, chat, kb, ws
from app.config import settings

logger = logging.getLogger(__name__)


async def _warm_up() -> None:
    """Preload lazy singletons so the first passenger turn is responsive.

    The sentiment pipeline and the embedding model load on first use; on a
    cold server that made the very first turn sit on "thinking" for tens of
    seconds. The DB pool is also connected here so the first request does not
    pay the asyncpg handshake. Warm-ups are best-effort: a failure is logged
    and the lazy path retries when actually needed, so startup never fails
    because of them.
    """
    from app.db import engine, init_models

    # Create/migrate schema (idempotent) and open one pool connection.
    try:
        await init_models()
        import sqlalchemy as sa

        async with engine.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
    except Exception:
        logger.exception("DB warm-up failed")

    from app.agents.retriever import get_retriever
    from app.agents.sentiment import get_sentiment_analyser

    async def warm_sentiment() -> None:
        try:
            await get_sentiment_analyser().analyse_async("warmup")
        except Exception:
            logger.exception("Sentiment model warm-up failed")

    async def warm_retrieval() -> None:
        try:
            await get_retriever().warm()
        except Exception:
            logger.exception("Retriever warm-up failed")

    await asyncio.gather(warm_sentiment(), warm_retrieval())


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _warm_up()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(agents.router)
app.include_router(ws.router)
app.include_router(kb.router)


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.app_name}


# Serve the built frontend from the same origin when present (production image).
# The frontend calls /api and opens WebSockets against location.host, so a
# single origin removes CORS and WS-proxy concerns. API/WS routes registered
# above win; this mount is the last-resort catch-all.
_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if (_dist / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
