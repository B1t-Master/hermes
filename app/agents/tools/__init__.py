"""LLM function calling for the hermes agent.

Tool schemas are OpenAI-compatible. Flight tools go through the
``hermes-flights`` MCP server (``app.agents.mcp_client``); rebooking routes
through the ``hermes-ops`` MCP server (with a direct fallback to
``services/rebook.py``). Booking/rebooking/tracking are the MCP-scoped
operations per the plan. Retrieval, profile lookups, baggage guidance, and
escalation stay in-process.

``run_tool_turn`` drives the model/tool loop: the model may call tools, results
are fed back as ``tool`` messages, and the loop returns once the model replies
with content instead of another tool call.
"""

from app.config import settings as settings
from services import email_real as email_real
from services import rebook as rebook

from .detection import (
    BOOKING_PATTERNS as BOOKING_PATTERNS,
)
from .detection import (
    REBOOK_PATTERNS as REBOOK_PATTERNS,
)
from .detection import (
    detect_booking_request,
    detect_rebook_request,
)
from .email import (
    _lookup_ticket_email as _lookup_ticket_email,
)
from .email import (
    _send_rebook_confirmation as _send_rebook_confirmation,
)
from .execution import (
    MAX_TOOL_ROUNDS,
    call_tool,
    run_tool_turn,
)
from .execution import (
    _canonical_name as _canonical_name,
)
from .execution import (
    _inject_booking_context as _inject_booking_context,
)
from .execution import (
    _parse_arguments as _parse_arguments,
)
from .specs import BOOKING_TOOL_SPECS, TOOL_SPECS

__all__ = [
    "BOOKING_TOOL_SPECS",
    "MAX_TOOL_ROUNDS",
    "TOOL_SPECS",
    "call_tool",
    "detect_booking_request",
    "detect_rebook_request",
    "run_tool_turn",
]
