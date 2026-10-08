"""Standalone MCP server for the mock KQ flight-delay service.

Run it directly (stdio transport, the default every MCP client speaks):

    .venv/bin/python -m services.flights_mcp

Or point an external MCP client at it, e.g. Claude Desktop:

    {
      "mcpServers": {
        "hermes-flights": {
          "command": "/path/to/hermes/.venv/bin/python",
          "args": ["-m", "services.flights_mcp"],
          "cwd": "/path/to/hermes"
        }
      }
    }

Tools exposed:

  - get_flight_status(flight_no, date?)   one flight's status on a day
  - list_delays(route?, date?, min_delay?)  delayed flights, worst first

The data is the same mock schedule the LangGraph `flight_lookup` node uses
in-process (`services/flight_delays.py`), so every consumer sees identical
answers for a given day.
"""

from mcp.server.fastmcp import FastMCP

from services import flight_delays

mcp = FastMCP(
    "hermes-flights",
    instructions=(
        "Mock Kenya Airways flight-status service. All schedules and delays "
        "are simulated prototype data, stable within a calendar day and "
        "regenerated each day. Flight numbers look like KQ720; routes use "
        "IATA codes such as NBO-LHR."
    ),
)


@mcp.tool()
def get_flight_status(flight_no: str, date: str | None = None) -> dict:
    """Status for one flight, e.g. "KQ720".

    Args:
        flight_no: flight number, with or without the "KQ" prefix.
        date: optional YYYY-MM-DD; defaults to today.
    """
    return flight_delays.get_flight_status(flight_no, date)


@mcp.tool()
def list_delays(route: str | None = None, date: str | None = None, min_delay: int = 15) -> dict:
    """Flights delayed by at least `min_delay` minutes, worst first.

    Returns {"delays": [...]} (a bare list would be interpreted by MCP as a
    list of content blocks, not a JSON payload).

    Args:
        route: optional "NBO-LHR" pair or single IATA code ("NBO"); omit for all.
        date: optional YYYY-MM-DD; defaults to today.
        min_delay: delay threshold in minutes (default 15).
    """
    return {"delays": flight_delays.list_delays(route=route, day=date, min_delay=min_delay)}


if __name__ == "__main__":
    mcp.run()  # stdio transport by default