"""Sentiment and intent classification nodes."""

import asyncio
import logging

from app.agents.intent import classify_intent
from app.agents.sentiment import SentimentResult, get_sentiment_analyser

from ..constants import SENTIMENT_TIMEOUT_SECONDS
from ..state import AgentState

logger = logging.getLogger(__name__)


async def analyse_sentiment(state: AgentState) -> dict:
    try:
        result = await asyncio.wait_for(
            get_sentiment_analyser().analyse_async(state.get("query", "")),
            timeout=SENTIMENT_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception("Sentiment step failed; assuming neutral")
        result = SentimentResult(label="neutral", score=0.0, confidence=0.0)
    return {"sentiment_score": result.score, "sentiment_label": result.label}


async def classify(state: AgentState) -> dict:
    result = await classify_intent(state.get("query", ""))
    return {"intent": result.intent, "intent_source": result.source}
