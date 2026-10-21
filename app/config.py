from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "hermes Agentic RAG"
    debug: bool = True
    sql_echo: bool = False
    secret_key: str = "change-me"
    access_token_expire_minutes: int = 60

    database_url: str = "postgresql+asyncpg://localhost:5432/hermes"

    # --- Database pool (Postgres only) ---
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_recycle: int = 1800
    db_pool_pre_ping: bool = True

    # --- Knowledge-base cache ---
    kb_cache_ttl_seconds: float = 300.0

    # --- Deployment / HTTP ---
    app_env: str = "development"
    cors_origins: str = "http://localhost:5173"

    groq_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_fallback_model: str = "openai/gpt-oss-20b"
    deepseek_model: str = "deepseek-chat"
    llm_timeout_seconds: float = 5.0
    llm_max_retries: int = 3
    # gpt-oss reasoning models: "low" cuts latency a lot on short support
    # answers. Empty string leaves the provider default untouched.
    llm_reasoning_effort: str = ""
    llm_max_answer_tokens: int = 512

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_escalation_threshold: float = -0.75
    sentiment_escalation_consecutive_turns: int = 2
    max_loop_turns: int = 3

    # --- Soft escalation (ask before handing off) ---
    # Hard triggers (safety, explicit "talk to a human") always escalate
    # immediately. The softer triggers (a negative-sentiment streak, or a
    # conversation the bot cannot close) first offer the passenger options and
    # ask. Set false to restore the old escalate-immediately behaviour.
    soft_escalation_enabled: bool = True
    # After this many unanswered offers, hand off to a human anyway so nobody
    # gets stuck in a loop.
    soft_escalation_max_offers: int = 2
    # Compensation/refund requests escalate immediately by default (money and
    # policy are involved). Set false to make them offer options first too.
    escalate_on_compensation: bool = True

    # --- Retrieval size (prompt token budget) ---
    # Lower k / a per-chunk cap shrink each request's prompt. Helpful on
    # providers with a low tokens-per-minute quota; higher values give the
    # model more to cite.
    retriever_k: int = 5
    context_max_chars: int = 0  # 0 disables the per-chunk cap

    agent_username: str = "agent"
    agent_password: str = "change-me"

    # --- Prototype outputs ---
    # Address seeded into rebook_email_map; confirms only go to mapped tickets.
    rebook_confirmation_email: str = "isaacmwanginjuguna@gmail.com"
    email_outbox_dir: str = ".mock_emails"

    # --- Email delivery (Resend) ---
    # With no API key the sender falls back to the mock outbox, so local runs
    # and tests need no credentials.
    resend_api_key: str = ""
    resend_from_email: str = "onboarding@resend.dev"

    # --- Telegram channel (long polling) ---
    # The adapter starts automatically when a token is present.
    telegram_bot_token: str = ""
    telegram_poll_timeout: int = 25
    telegram_api_base: str = "https://api.telegram.org"

    # --- Rebooking: official self-service page (path 1 of 2) ---
    # Surfaced alongside the in-chat rebooking so the passenger can choose.
    self_service_rebooking_url: str = (
        "https://digital.kenya-airways.com/servicing/self-reaccommodation/retrieve"
    )

    # --- Baggage guidance: lost / delayed / unclaimed bags ---
    # Deterministic, in-process content (not LLM-invented). Editable per station.
    baggage_arrival_claim: str = "JKIA Terminal 1A - Arrivals baggage claim hall"
    baggage_lost_and_found: str = (
        "JKIA Terminal 1A - Airport Lost & Found desk (Arrivals level)"
    )
    worldtracer_url: str = "https://www.worldtracer.aero/"

    # --- Flight MCP client ---
    # Answer flight status/delay questions through the hermes-flights MCP
    # server (stdio). Set false to call the in-process library directly; the
    # client also falls back on its own when the server is unreachable.
    flights_mcp_enabled: bool = True

    @field_validator("database_url", mode="before")
    @classmethod
    def _force_asyncpg_driver(cls, value: object) -> object:
        """Normalise Postgres DSNs to what the asyncpg driver expects.

        Neon (and Heroku) hand out ``postgres://`` / ``postgresql://`` URLs, but
        the engine is async (``create_async_engine``) and asyncpg is the driver
        this project installs. Without the explicit ``+asyncpg`` SQLAlchemy
        falls back to psycopg2, which is not a dependency, so every import of
        ``app.db`` dies with ``ModuleNotFoundError: No module named 'psycopg2'``
        and ingest's ``auto`` store check silently misses the pg backend.

        asyncpg also rejects libpq-only query params: ``sslmode`` becomes
        ``ssl`` (asyncpg accepts the same mode strings) and ``channel_binding``
        is dropped. Without this, the connection fails with
        ``TypeError: connect() got an unexpected keyword argument 'sslmode'``.
        """
        if not isinstance(value, str):
            return value
        for prefix in ("postgres://", "postgresql://"):
            if value.startswith(prefix):
                value = "postgresql+asyncpg://" + value[len(prefix):]
                break
        if not value.startswith("postgresql+asyncpg://"):
            return value
        parts = urlsplit(value)
        translated: list[tuple[str, str]] = []
        for key, val in parse_qsl(parts.query, keep_blank_values=True):
            if key == "channel_binding":
                continue
            translated.append(("ssl" if key == "sslmode" else key, val))
        if translated == parse_qsl(parts.query, keep_blank_values=True):
            return value
        return urlunsplit(parts._replace(query=urlencode(translated)))

    @property
    def is_llm_configured(self) -> bool:
        return bool(self.groq_api_key or self.deepseek_api_key)

    @property
    def telegram_enabled(self) -> bool:
        """The adapter starts only when a bot token is configured."""
        return bool(self.telegram_bot_token)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
