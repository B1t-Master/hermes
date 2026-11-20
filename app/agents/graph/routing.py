"""Conditional-edge selectors for the conversation graph."""

from .state import AgentState


def dispatch_next(state: AgentState) -> str:
    return "greeting" if state.get("greeting") else "chat"


def route_next(state: AgentState) -> str:
    """Route to the human, or on to the answer path.

    A soft trigger no longer diverts the turn: the answer is produced first
    and the hand-off is offered as a quick reply beside it.
    """
    if state.get("escalate"):
        return "escalate"
    return "retrieve"
