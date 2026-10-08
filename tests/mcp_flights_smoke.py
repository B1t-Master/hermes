"""Smoke test for the mock flight-delay MCP server.

Verifies three contracts:

  1. The FastMCP server answers through a real MCP stdio client session
     (`ClientSession`), the transport any external MCP client would use.
  2. Determinism: repeated lookups for the same date return identical data.
  3. Consistency: the in-process functions the graph uses
     (`services.flight_delays`) return exactly what the MCP tools return.

Run with:  .venv/bin/python tests/mcp_flights_smoke.py
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services import flight_delays

DATE = "2026-10-08"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f"  [{detail}]" if detail else ""), flush=True)


async def call_tool(session: ClientSession, name: str, args: dict):
    result = await session.call_tool(name, args)
    if result.isError:
        raise RuntimeError(f"tool {name} error: {result.content}")
    if not result.content:
        # FastMCP sends no content block for an empty-list result (e.g. no
        # delays on a quiet route).
        return []
    return json.loads(result.content[0].text)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "flights.sqlite")
        env = {**os.environ, "FLIGHTS_DB_PATH": db}
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "services.flights_mcp"],
            env=env,
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Tool listing
                tools = {t.name for t in (await session.list_tools()).tools}
                check("MCP lists both tools", tools == {"get_flight_status", "list_delays"}, ",".join(sorted(tools)))

                # Status lookup through MCP
                mcp_status = await call_tool(session, "get_flight_status", {"flight_no": "KQ720", "date": DATE})
                check("MCP get_flight_status found KQ720", mcp_status.get("found") is True, f"status={mcp_status.get('status')}")
                check("MCP status shape", all(k in mcp_status for k in ("flight_no", "route", "status", "delay_minutes", "scheduled_departure")), mcp_status.get("route", ""))

                # Determinism: same date twice -> identical
                again = await call_tool(session, "get_flight_status", {"flight_no": "KQ720", "date": DATE})
                check("deterministic within a day", again == mcp_status, f"delay={again.get('delay_minutes')}")

                # Delays listing: KQ116 (NBO-FRA) is delayed 25 min on the
                # pinned date (2026-10-08), so this exercises a non-empty path.
                delays_payload = await call_tool(session, "list_delays", {"route": "NBO-FRA", "date": DATE, "min_delay": 15})
                delays = delays_payload["delays"]
                check("list_delays returns a dict payload", isinstance(delays, list), f"{len(delays)} delayed")
                for row in delays:
                    check("delay row shape", all(k in row for k in ("flight_no", "delay_minutes", "status")), row.get("flight_no", ""))
                    break  # cover first row only

                # Quiet route -> empty list in a JSON payload
                quiet = (await call_tool(session, "list_delays", {"route": "NBO-LHR", "date": DATE, "min_delay": 15}))["delays"]
                check("quiet route returns []", quiet == [], f"{len(quiet)} delayed")

                # Unknown flight -> clean not-found shape
                missing = await call_tool(session, "get_flight_status", {"flight_no": "KQ999", "date": DATE})
                check("unknown flight returns found=False", missing.get("found") is False, missing.get("message", ""))

        # Consistency: in-process functions agree with the MCP responses
        local_status = flight_delays.get_flight_status("KQ720", DATE, db_path=db)
        check("in-process == MCP status", local_status == mcp_status, f"delay={local_status.get('delay_minutes')}")
        local_delays = flight_delays.list_delays("NBO-FRA", DATE, min_delay=15, db_path=db)
        check("in-process == MCP delays", local_delays == delays, f"{len(local_delays)} delayed")

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())