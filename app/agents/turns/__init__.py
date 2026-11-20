"""One passenger turn = persist input -> run graph -> persist outputs.

Shared by the WebSocket handler (streaming) and the REST fallback
(non-streaming), so there is exactly one pipeline to reason about.

This package splits the original module into text cleanup (``text``), the
conversation persistence helpers (``store``), and the turn pipeline
(``orchestration``). Public names are re-exported here so existing imports
keep working unchanged.
"""

from .orchestration import run_turn
from .store import (
    HISTORY_LIMIT,
    booking_to_dict,
    get_conversation,
    load_history,
    load_profile,
    message_to_dict,
)
from .text import (
    citations_from_chunks,
    strip_source_section,
    strip_source_urls,
)

__all__ = [
    "HISTORY_LIMIT",
    "booking_to_dict",
    "citations_from_chunks",
    "get_conversation",
    "load_history",
    "load_profile",
    "message_to_dict",
    "run_turn",
    "strip_source_section",
    "strip_source_urls",
]
