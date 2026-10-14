"""Best-effort performance metrics.

Turns append rows to ``performance_metrics`` for the agent dashboard's
Metrics tab: how long the bot took, how many chunks retrieval returned, whether
a turn escalated, and the passenger's end-of-conversation rating.

Metrics are observability, never part of the answer path. An unknown metric
name is ignored rather than raising.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import PerformanceMetric

logger = logging.getLogger(__name__)

RESPONSE_TIME_MS = "response_time_ms"
RETRIEVAL_HITS = "retrieval_hits"
ESCALATION = "escalation"
CSAT = "csat"

METRIC_NAMES = frozenset({RESPONSE_TIME_MS, RETRIEVAL_HITS, ESCALATION, CSAT})


def add_metric(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    metric: str,
    value: float,
) -> None:
    """Queue a metric row on an open session (its caller commits)."""
    if metric not in METRIC_NAMES:
        logger.warning("ignoring unknown metric %r", metric)
        return
    session.add(
        PerformanceMetric(
            conversation_id=conversation_id,
            metric=metric,
            value=float(value),
        )
    )
