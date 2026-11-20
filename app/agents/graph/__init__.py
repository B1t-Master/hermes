"""LangGraph orchestration: greeting -> sentiment -> intent -> router -> QA | escalation.

Flow:

    dispatch -> analyse_sentiment -> classify -> route
        route --(escalate)--> handle_escalation --> END
        route --(routine)--> retrieve --> flight_lookup --> baggage_lookup --> generate_answer --> END

A standalone greeting ("hi") is answered by the `greeting` node and never
reaches retrieval: the knowledge base has nothing to say about it, and
treating "no context" as "cannot help" made the first turn of every
conversation end in a human handoff.

Escalation triggers (in `route`):
  - hard: intent in {safety, human_request}. Keyword safety hits arrive here
    via the intent layer's hard-trigger path. A hard trigger still gets a bot
    message (the acknowledgement), never silence.
  - soft: intent in {compensation}, sentiment below
    `sentiment_escalation_threshold` for `sentiment_escalation_consecutive_turns`
    consecutive passenger turns, or a conversation running past `max_loop_turns`
    with a non-routine intent. A soft trigger does NOT replace the answer: the
    bot answers the passenger and adds a "Connect me to a human" quick reply.
    Only after `soft_escalation_max_offers` unanswered offers does it hand off.

Streaming: the caller may pass `on_token` (async callable) through
`config["configurable"]`. The QA node then streams LLM tokens as they arrive
instead of blocking for the full answer.
"""

from app.agents.baggage import detect_baggage_request as detect_baggage_request
from app.config import settings as settings

from .builder import build_graph, get_graph
from .constants import (
    BAGGAGE_INSTRUCTION as BAGGAGE_INSTRUCTION,
)
from .constants import (
    BAGGAGE_OPTIONS as BAGGAGE_OPTIONS,
)
from .constants import (
    BOOKING_TOOL_INSTRUCTION as BOOKING_TOOL_INSTRUCTION,
)
from .constants import (
    ESCALATION_ACK as ESCALATION_ACK,
)
from .constants import (
    ESCALATION_ACK_HUMAN as ESCALATION_ACK_HUMAN,
)
from .constants import (
    ESCALATION_ACK_SAFETY as ESCALATION_ACK_SAFETY,
)
from .constants import (
    HARD_ESCALATING_INTENTS as HARD_ESCALATING_INTENTS,
)
from .constants import (
    HISTORY_TURNS_FOR_CONTEXT as HISTORY_TURNS_FOR_CONTEXT,
)
from .constants import (
    OFFER_ESCALATION_TEXT as OFFER_ESCALATION_TEXT,
)
from .constants import (
    QA_SYSTEM_PROMPT as QA_SYSTEM_PROMPT,
)
from .constants import (
    REBOOK_TOOL_INSTRUCTION as REBOOK_TOOL_INSTRUCTION,
)
from .constants import (
    SENTIMENT_TIMEOUT_SECONDS as SENTIMENT_TIMEOUT_SECONDS,
)
from .constants import (
    SOFT_ESCALATING_INTENTS as SOFT_ESCALATING_INTENTS,
)
from .constants import (
    SOFT_ESCALATION_OPTIONS as SOFT_ESCALATION_OPTIONS,
)
from .constants import (
    SOFT_OFFER_INSTRUCTION as SOFT_OFFER_INSTRUCTION,
)
from .constants import (
    SUMMARY_MESSAGES_FOR_ESCALATION as SUMMARY_MESSAGES_FOR_ESCALATION,
)
from .nodes import (
    analyse_sentiment,
    answer,
    baggage_lookup,
    classify,
    dispatch,
    escalate,
    flight_lookup,
    greet,
    retrieve,
    route,
)
from .nodes.answer import _known_context_block as _known_context_block
from .nodes.escalation import _escalation_ack as _escalation_ack
from .nodes.retrieval import _format_context as _format_context
from .nodes.route import _history_triggers as _history_triggers
from .routing import dispatch_next as _dispatch_next  # noqa: F401
from .routing import route_next as _route_next  # noqa: F401
from .state import AgentState

__all__ = [
    "AgentState",
    "analyse_sentiment",
    "answer",
    "baggage_lookup",
    "build_graph",
    "classify",
    "dispatch",
    "escalate",
    "flight_lookup",
    "get_graph",
    "greet",
    "retrieve",
    "route",
]
