"""Greeting node: answer standalone small talk before retrieval."""

from app.agents.greeting import greeting_reply

from ..state import AgentState


async def dispatch(state: AgentState) -> dict:
    """Catch small talk before the retrieval pipeline sees it.

    "Hi" has no answer in the knowledge base, and the QA prompt treats a thin
    context as "cannot help" — which used to greet every new conversation with
    a hand-off offer. Greetings are answered deterministically instead.
    """
    reply = greeting_reply(state.get("query", ""))
    if reply is None:
        return {}
    text, options = reply
    return {
        "greeting": "greeting",
        "answer": text,
        "escalation_offer": {"reason": "greeting", "options": options},
    }


async def greet(state: AgentState) -> dict:
    """Terminal node for a greeting turn: the answer is already set."""
    return {}
