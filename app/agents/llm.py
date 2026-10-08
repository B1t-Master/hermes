"""Single LLM entry point for the whole backend.

Fallback chain, cheapest resilience first:
  1. Groq primary model (`llm_model`)
  2. Groq small model (`llm_fallback_model`)
  3. DeepSeek (`deepseek_model`), if a key is configured

`complete()` walks the chain on any error. `stream()` tries providers in
order for the initial request; a mid-stream failure raises `LLMError` and
the caller decides what to tell the user (restarting would duplicate tokens).
"""

import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMError(RuntimeError):
    """Every provider in the chain failed."""


class LLM:
    def __init__(self) -> None:
        self._chain_cache: list[tuple[AsyncOpenAI, str]] | None = None

    @property
    def is_configured(self) -> bool:
        return settings.is_llm_configured

    def _build_chain(self) -> list[tuple[AsyncOpenAI, str]]:
        chain: list[tuple[AsyncOpenAI, str]] = []
        common = {
            "timeout": settings.llm_timeout_seconds,
            "max_retries": settings.llm_max_retries,
        }
        if settings.groq_api_key:
            groq = AsyncOpenAI(api_key=settings.groq_api_key, base_url=GROQ_BASE_URL, **common)
            chain.append((groq, settings.llm_model))
            if settings.llm_fallback_model:
                chain.append((groq, settings.llm_fallback_model))
        if settings.deepseek_api_key:
            deepseek = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                **common,
            )
            chain.append((deepseek, settings.deepseek_model))
        return chain

    def _chain(self) -> list[tuple[AsyncOpenAI, str]]:
        if self._chain_cache is None:
            self._chain_cache = self._build_chain()
        return self._chain_cache

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        chain = self._chain()
        if not chain:
            raise LLMError("No LLM provider configured (set GROQ_API_KEY or DEEPSEEK_API_KEY)")
        errors: list[str] = []
        for client, model in chain:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if content:
                    return content
                errors.append(f"{model}: empty response")
            except Exception as exc:
                errors.append(f"{model}: {type(exc).__name__} {exc}")
                logger.warning("LLM attempt failed (%s): %s", model, exc)
        raise LLMError("; ".join(errors))

    async def stream(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        chain = self._chain()
        if not chain:
            raise LLMError("No LLM provider configured (set GROQ_API_KEY or DEEPSEEK_API_KEY)")

        last_error: Exception | None = None
        for client, model in chain:
            try:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
                return
            except Exception as exc:
                last_error = exc
                logger.warning("LLM stream attempt failed (%s): %s", model, exc)
        raise LLMError(f"All LLM providers failed: {last_error}")


_llm: LLM | None = None


def get_llm() -> LLM:
    """Shared instance: one client pool, one fallback chain."""
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm
