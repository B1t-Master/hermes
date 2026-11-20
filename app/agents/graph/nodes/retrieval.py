"""Retrieval and context-enrichment nodes."""

import logging

from app.agents.baggage import render_guidance
from app.agents.flights import detect_flight_request, get_flight_lookup
from app.agents.retriever import get_retriever
from app.config import settings

from ..state import AgentState

logger = logging.getLogger(__name__)


async def retrieve(state: AgentState) -> dict:
    chunks = await get_retriever().retrieve(state.get("query", ""))
    return {"chunks": [chunk.__dict__ for chunk in chunks]}


def _format_context(chunks: list[dict]) -> tuple[str, str]:
    cap = settings.context_max_chars
    parts, titles = [], []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title") or chunk.get("source_url") or f"source {i}"
        titles.append(title)
        content = chunk.get("content", "")
        if cap and len(content) > cap:
            content = content[:cap].rstrip() + "…"
        parts.append(f"[{i}] {title}\n{content}")
    return "\n\n".join(parts), ", ".join(dict.fromkeys(titles)) or "none"


async def flight_lookup(state: AgentState) -> dict:
    """Fetch mock flight status/delays when the query asks about a flight.

    Detection is deterministic (`detect_flight_request`), so the node only
    runs pipeline work on genuine flight questions; everything else returns
    an empty update and the graph continues unchanged. Failures degrade to
    "no flight context" rather than failing the turn.
    """
    request = detect_flight_request(state.get("query", ""))
    if request is None:
        return {}
    lookup = get_flight_lookup()
    try:
        if request["type"] == "status":
            info = await lookup.status(request["flight_no"])
            context = lookup.render_status(info)
        else:
            rows = await lookup.delays(request.get("route"))
            context = lookup.render_delays(rows)
            if not context:
                context = (
                    f"No flights on {request.get('route') or 'those routes'} "
                    "are delayed right now."
                )
    except Exception:
        logger.exception("Flight lookup failed; continuing without flight context")
        return {}
    return {"flight_context": context}


async def baggage_lookup(state: AgentState) -> dict:
    """Attach lost/delayed-bag guidance when the query is about a missing bag.

    Deterministic detection and in-process content: no external call, no KB
    lookup needed. The QA node relays the three options and offers a human.
    """
    # Resolve through the package namespace so the historical
    # ``app.agents.graph.detect_baggage_request`` patch point keeps working.
    from app.agents import graph as graph_pkg

    if not graph_pkg.detect_baggage_request(state.get("query", "")):
        return {}
    return {"baggage_context": render_guidance()}
