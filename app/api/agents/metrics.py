"""Aggregated turn/queue metrics for the dashboard Metrics tab."""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps import get_current_agent
from app.metrics import CSAT, ESCALATION, RESPONSE_TIME_MS, RETRIEVAL_HITS
from app.models import Agent, PerformanceMetric
from app.schemas import MetricsOut

from .common import _mean, router


@router.get("/metrics", response_model=MetricsOut)
async def metrics(
    _: Agent = Depends(get_current_agent),
    db: AsyncSession = Depends(get_db),
):
    """Aggregated turn/queue metrics for the dashboard Metrics tab."""
    pm = PerformanceMetric

    response_rows = (
        await db.execute(select(pm.value).where(pm.metric == RESPONSE_TIME_MS))
    ).scalars().all()
    retrieval_rows = (
        await db.execute(select(pm.value).where(pm.metric == RETRIEVAL_HITS))
    ).scalars().all()
    csat_rows = (
        await db.execute(select(pm.value).where(pm.metric == CSAT))
    ).scalars().all()
    escalations = (
        await db.execute(
            select(func.count()).select_from(pm).where(pm.metric == ESCALATION)
        )
    ).scalar_one()

    return MetricsOut(
        avg_response_ms=_mean(response_rows),
        avg_retrieval_hits=_mean(retrieval_rows),
        escalations=int(escalations),
        csat_responses=len(csat_rows),
        avg_csat=_mean(csat_rows),
    )
