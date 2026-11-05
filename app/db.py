import uuid
from datetime import date, timedelta

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
        # conversations.soft_offers: counts soft-escalation offers already made.
        has_soft_offers = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_name = 'conversations' AND column_name = 'soft_offers'"
                )
            )
        ).scalar()
        if not has_soft_offers:
            await conn.execute(
                text(
                    "ALTER TABLE conversations "
                    "ADD COLUMN soft_offers INTEGER NOT NULL DEFAULT 0"
                )
            )
        # messages.actions: quick-reply buttons on a bot message (JSON list).
        has_actions = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_name = 'messages' AND column_name = 'actions'"
                )
            )
        ).scalar()
        if not has_actions:
            await conn.execute(text("ALTER TABLE messages ADD COLUMN actions JSON"))
        # messages.payment_id: the in-chat payment card belongs to a bot
        # message. Added after the table first existed, so guard it explicitly.
        has_payment_id = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_name = 'messages' AND column_name = 'payment_id'"
                )
            )
        ).scalar()
        if not has_payment_id:
            await conn.execute(
                text(
                    "ALTER TABLE messages ADD COLUMN payment_id UUID "
                    "REFERENCES payments(id) ON DELETE SET NULL"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_messages_payment_id "
                    "ON messages (payment_id)"
                )
            )
        # conversations.facts: structured memory slots (JSON object).
        has_facts = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_name = 'conversations' AND column_name = 'facts'"
                )
            )
        ).scalar()
        if not has_facts:
            await conn.execute(text("ALTER TABLE conversations ADD COLUMN facts JSON"))
        # bookings.passenger_email was made nullable so a booking can be held
        # before the passenger supplies their email (payment happens first).
        has_email_nullable = (
            await conn.execute(
                text(
                    "SELECT is_nullable FROM information_schema.columns "
                    "WHERE table_name = 'bookings' AND column_name = 'passenger_email'"
                )
            )
        ).scalar()
        if has_email_nullable == "NO":
            await conn.execute(
                text("ALTER TABLE bookings ALTER COLUMN passenger_email DROP NOT NULL")
            )
        # Ticket -> confirmation email. Seeded with prototype rows; ON CONFLICT
        # keeps this idempotent and never clobbers rows added later.
        seed_email = settings.rebook_confirmation_email
        for ticket in ("HRM-9B4770", "KQ-123456", "1234567890", "TEST-0001"):
            await conn.execute(
                text(
                    "INSERT INTO rebook_email_map (ticket_number, email) "
                    "VALUES (:ticket, :email) "
                    "ON CONFLICT (ticket_number) DO NOTHING"
                ),
                {"ticket": ticket, "email": seed_email},
            )
        # Prototype booking(s) so identity inheritance is demonstrable: a
        # passenger logging in with the seed email sees their upcoming trip.
        depart = date.today() + timedelta(days=14)
        await conn.execute(
            text(
                "INSERT INTO bookings "
                "(id, pnr, passenger_email, passenger_name, flight_no, origin, "
                " destination, depart_date, depart_time, status, currency, amount) "
                "VALUES (:id, :pnr, :email, :name, :flight, :origin, :dest, "
                " :date, :time, 'ticketed', 'KES', :amount) "
                "ON CONFLICT (pnr) DO NOTHING"
            ),
            {
                "id": uuid.uuid4(),
                "pnr": "KQ7X2B9",
                "email": seed_email,
                "name": "Isaac Mwangi",
                "flight": "KQ100",
                "origin": "NBO",
                "dest": "LHR",
                "date": depart.isoformat(),
                "time": "23:59",
                "amount": 78500.0,
            },
        )
