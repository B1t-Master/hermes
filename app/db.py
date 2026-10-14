from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    pass


def _engine_kwargs() -> dict:
    """Engine options for Postgres (asyncpg)."""
    return {
        "echo": settings.sql_echo,
        "pool_pre_ping": settings.db_pool_pre_ping,
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_recycle": settings.db_pool_recycle,
    }


engine = create_async_engine(settings.database_url, **_engine_kwargs())

SessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    async with SessionLocal() as session:
        yield session


async def init_models() -> None:
    """Create schema / tables.

    Runs 'CREATE EXTENSION IF NOT EXISTS vector;' automatically so a
    manual Neon step is not required, plus guards for schema drift on
    tables that predate the current model set.
    """
    import app.models  # noqa: F401  (register models on Base)

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        nullable = (
            await conn.execute(
                text(
                    "SELECT is_nullable FROM information_schema.columns "
                    "WHERE table_name = 'knowledge_fragments' AND column_name = 'embedding'"
                )
            )
        ).scalar()
        if nullable == "NO":
            await conn.execute(
                text(
                    "ALTER TABLE knowledge_fragments ALTER COLUMN embedding DROP NOT NULL")
            )
        # Message citations: `messages.sources` was added after the table first
        # existed. create_all does not alter existing tables, so add the
        # column explicitly.
        has_sources = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_name = 'messages' AND column_name = 'sources'"
                )
            )
        ).scalar()
        if not has_sources:
            await conn.execute(text("ALTER TABLE messages ADD COLUMN sources JSON"))
        # passengers.telegram_chat_id links a Telegram chat to a passenger.
        # Same drift guard: create_all will not alter the pre-existing table.
        has_telegram = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_name = 'passengers' AND column_name = 'telegram_chat_id'"
                )
            )
        ).scalar()
        if not has_telegram:
            await conn.execute(
                text("ALTER TABLE passengers ADD COLUMN telegram_chat_id VARCHAR(64)")
            )
            await conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_passengers_telegram_chat_id "
                    "ON passengers (telegram_chat_id)"
                )
            )
