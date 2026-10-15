"""Flight lookup for the LangGraph pipeline, backed by the MCP server.

The graph queries the mock flight service through the `hermes-flights` MCP
server (`app.agents.mcp_client`), which falls back to the in-process library
(`services/flight_delays.py`) when the server is unreachable. Detection is
deterministic (regex + keyword), not LLM-driven, so the decision to look up a
flight is auditable and cheap; the LLM only ever sees the rendered result via
the `flight_context` state field.
"""

from __future__ import annotations

import re

from app.agents.mcp_client import get_flights_client
from services import flight_delays

FLIGHT_NO_RE = re.compile(r"\bKQ[\s\-]*(\d{2,4})\b", re.IGNORECASE)

STATUS_KEYWORDS = (
    "status",
    "delay",
    "delayed",
    "delays",
    "on time",
    "on-time",
    "schedule",
    "scheduled",
    "depart",
    "arrival",
    "arrive",
    "cancelled",
    "cancel",
    "boarding",
    "gate",
    "flight",
    "flying",
)

# Known IATA codes (the mock schedule's airports) plus city aliases.
IATA_CODES = {
    "NBO": "NBO", "LHR": "LHR", "CDG": "CDG", "FRA": "FRA", "DXB": "DXB",
    "JNB": "JNB", "CPT": "CPT", "DAR": "DAR", "MBA": "MBA", "EBB": "EBB",
    "CMB": "CMB", "DOH": "DOH", "ACC": "ACC",
}
CITY_TO_CODE = {
    "nairobi": "NBO", "london": "LHR", "paris": "CDG", "frankfurt": "FRA",
    "dubai": "DXB", "johannesburg": "JNB", "cape town": "CPT",
    "dar es salaam": "DAR", "mombasa": "MBA", "entebbe": "EBB",
    "colombo": "CMB", "doha": "DOH", "accra": "ACC",
}


def detect_flight_request(query: str) -> dict | None:
    """Decide whether `query` asks about flight status or delays.

    Returns {"type": "status", "flight_no": "KQ720"} for flight-number
    questions, {"type": "delays", "route": "NBO-LHR" | "NBO" | None} for
    route-level delay questions, or None when the query is not flight-related.
    """
    match = FLIGHT_NO_RE.search(query)
    if match:
        return {"type": "status", "flight_no": f"KQ{match.group(1)}"}

    lower = query.lower()
    if not any(word in lower for word in STATUS_KEYWORDS):
        return None

    # Collect (position, code) from direct IATA matches and city aliases, so
    # route ordering follows the order they appear in the query.
    upper = query.upper()
    positions: list[tuple[int, str]] = []
    for code in IATA_CODES:
        match = re.search(rf"\b{code}\b", upper)
        if match:
            positions.append((match.start(), code))
    for city, code in CITY_TO_CODE.items():
        idx = lower.find(city)
        if idx != -1:
            positions.append((idx, code))

    if not positions:
        return None
    positions.sort()
    seen: set[str] = set()
    ordered: list[str] = []
    for _, code in positions:
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    if len(ordered) >= 2:
        return {"type": "delays", "route": f"{ordered[0]}-{ordered[1]}"}
    return {"type": "delays", "route": ordered[0]}


class FlightLookup:
    """Async facade over the flight MCP client (SQLite stays off the loop)."""

    async def status(self, flight_no: str, day: str | None = None) -> dict:
        return await get_flights_client().status(flight_no, day)

    async def delays(self, route: str | None = None, day: str | None = None) -> list[dict]:
        return await get_flights_client().delays(route, day)

    @staticmethod
    def render_status(info: dict) -> str:
        if not info.get("found"):
            return f"No flight {info.get('flight_no', '?')} in the mock schedule."
        line = (
            f"{info['flight_no']} ({info['route']}, scheduled "
            f"{info['scheduled_departure']}->{info['scheduled_arrival']} local): {info['status']}"
        )
        if info.get("status") == flight_delays.STATUS_DELAYED:
            line += f", delayed by {info['delay_minutes']} minutes"
        return line

    @staticmethod
    def render_delays(rows: list[dict]) -> str:
        if not rows:
            return ""
        parts = [f"{r['flight_no']} {r['route']} (+{r['delay_minutes']} min)" for r in rows]
        return "Delayed flights: " + "; ".join(parts)


_lookup: FlightLookup | None = None


def get_flight_lookup() -> FlightLookup:
    global _lookup
    if _lookup is None:
        _lookup = FlightLookup()
    return _lookup