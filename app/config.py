from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "hermes Agentic RAG"
    debug: bool = True
    sql_echo: bool = False
    secret_key: str = "change-me"
    access_token_expire_minutes: int = 60

    database_url: str = "postgresql+asyncpg://localhost:5432/hermes"

    groq_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_fallback_model: str = "openai/gpt-oss-20b"
    deepseek_model: str = "deepseek-chat"
    llm_timeout_seconds: float = 5.0
    llm_max_retries: int = 3

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_escalation_threshold: float = -0.75
    sentiment_escalation_consecutive_turns: int = 2
    max_loop_turns: int = 3

    agent_username: str = "agent"
    agent_password: str = "change-me"

    @property
    def is_llm_configured(self) -> bool:
        return bool(self.groq_api_key or self.deepseek_api_key)


settings = Settings()