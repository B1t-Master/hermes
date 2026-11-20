"""Shared router, helpers, and broadcasts for the agent dashboard API."""

from __future__ import annotations

from fastapi import APIRouter

from app.agents.hub import dashboard_hub
from app.models import Escalation

router = APIRouter(prefix="/agents", tags=["agents"])

_STATUS_VALUES = ("active", "pending", "assigned", "resolved", "all")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


async def _broadcast(escalation: Escalation, action: str) -> None:
    await dashboard_hub.broadcast(
        {
            "type": "queue_update",
            "action": action,
            "escalation_id": str(escalation.id),
            "conversation_id": str(escalation.conversation_id),
        }
    )
