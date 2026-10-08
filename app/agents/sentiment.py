"""Passenger message sentiment: local CPU classifier, lazily loaded.

`cardiffnlp/twitter-roberta-base-sentiment-latest` emits probabilities over
{negative, neutral, positive}. We expose a signed score in [-1, 1]:

    score = P(positive) - P(negative)

so the config threshold (`sentiment_escalation_threshold = -0.75`) reads
naturally: strongly negative sentiment. Neutral messages land near 0.
"""

import asyncio
import logging
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    label: str  # negative | neutral | positive
    score: float  # signed, [-1, 1]
    confidence: float  # probability of the argmax label


class SentimentAnalyser:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.sentiment_model
        self._pipeline = None

    @property
    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        from transformers import pipeline

        logger.info("Loading sentiment model %s (CPU)", self.model_name)
        self._pipeline = pipeline(
            task="text-classification",
            model=self.model_name,
            top_k=None,  # all labels, not just the max
            truncation=True,
        )
        return self._pipeline

    def analyse(self, text: str) -> SentimentResult:
        """Sync analysis; callers on the event loop should use `analyse_async`."""
        cleaned = " ".join((text or "").split())  # collapse whitespace/newlines
        if not cleaned:
            return SentimentResult(label="neutral", score=0.0, confidence=0.0)
        if not self.available:
            logger.warning("transformers not installed; neutral sentiment assumed")
            return SentimentResult(label="neutral", score=0.0, confidence=0.0)

        try:
            pipeline = self._load()
            predictions = pipeline(cleaned[:1024])[0]  # list of {label, score}
        except Exception:
            # Model download/load or inference failure must not kill the
            # answer path: degrade to neutral instead.
            logger.exception("Sentiment inference failed; assuming neutral")
            return SentimentResult(label="neutral", score=0.0, confidence=0.0)
        by_label = {p["label"].lower(): float(p["score"]) for p in predictions}

        label = max(by_label, key=by_label.get)
        signed = by_label.get("positive", 0.0) - by_label.get("negative", 0.0)
        return SentimentResult(
            label=label,
            score=round(signed, 4),
            confidence=round(by_label[label], 4),
        )

    async def analyse_async(self, text: str) -> SentimentResult:
        """Run the model off the event loop (CPU inference blocks)."""
        return await asyncio.to_thread(self.analyse, text)


_analyser: SentimentAnalyser | None = None


def get_sentiment_analyser() -> SentimentAnalyser:
    """Shared instance: the model loads at most once."""
    global _analyser
    if _analyser is None:
        _analyser = SentimentAnalyser()
    return _analyser
