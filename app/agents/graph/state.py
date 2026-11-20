"""Shared LangGraph state type."""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    conversation_id: str
    passenger_id: str
    query: str
    history: list[dict]  # [{role, content}, ...] most recent last
    conversation_status: str  # pre-loaded by run_turn (avoids a re-fetch)
    conversation_turn_count: int
    sentiment_score: float
    sentiment_label: str
    intent: str
    intent_source: str
    escalate: bool
    escalation_reason: str
    escalation_created: bool
    # Soft-escalation: offer options instead of escalating outright. Set by
    # `route`; the answer node attaches the quick replies to its answer.
    offer_escalation: bool
    escalation_offer: dict
    soft_offers: int  # offers already made in this conversation
    chunks: list[dict]  # RetrievedChunk as plain dicts
    flight_context: str  # rendered mock flight status/delays, when relevant
    baggage_context: str  # rendered lost/delayed-bag guidance, when relevant
    facts: dict  # persisted conversation memory slots (ticket, phone, ...)
    profile: dict  # passenger identity + bookings, for inheritance
    greeting: str  # "greeting" | "thanks" when the turn is small talk
    tool_results: list[dict]  # raw payloads returned by the tools this turn
    answer: str
