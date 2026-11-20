"""Integration test: the bot's MCP client talks to the real flights server.

Unlike `tests/mcp_flights_smoke.py` (a script), this runs under pytest. It
starts `services.flights_mcp` over stdio through `FlightsMCPClient`, calls the
tools directly (no fallback), and checks the in-process and MCP answers agree.

Run with:  .venv/bin/python -m pytest tests/test_mcp_client.py -q
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.agents import mcp_client
from app.agents.mcp_client import FlightsMCPClient
from services import flight_delays

DATE = "2026-10-08"


class FlightsMCPClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._db = str(Path(self._tmp.name) / "flights.sqlite")
        self._env = mock.patch.dict(os.environ, {"FLIGHTS_DB_PATH": self._db})
        self._env.start()
        self.client = FlightsMCPClient()

    async def asyncTearDown(self):
        await self.client.stop()
        self._env.stop()
        self._tmp.cleanup()

    async def test_get_flight_status_over_mcp(self):
        # call_tool does not fall back, so this fails if the server is broken.
        payload = await self.client.call_tool(
            "get_flight_status", {"flight_no": "KQ720", "date": DATE}
        )
        self.assertTrue(payload["found"])
        self.assertEqual(payload["flight_no"], "KQ720")
        self.assertIn("route", payload)
        self.assertIn("scheduled_departure", payload)

    async def test_list_delays_over_mcp(self):
        payload = await self.client.call_tool("list_delays", {"date": DATE})
        self.assertIsInstance(payload, dict)
        self.assertIsInstance(payload["delays"], list)
        for row in payload["delays"]:
            self.assertGreaterEqual(row["delay_minutes"], 15)
            break

    async def test_status_method_matches_in_process(self):
        with mock.patch.object(mcp_client.settings, "flights_mcp_enabled", True):
            over_mcp = await self.client.status("KQ720", DATE)
        in_process = flight_delays.get_flight_status("KQ720", DATE, db_path=self._db)
        self.assertEqual(over_mcp, in_process)

    async def test_unknown_flight_is_clean(self):
        payload = await self.client.call_tool(
            "get_flight_status", {"flight_no": "KQ999", "date": DATE}
        )
        self.assertFalse(payload["found"])

    async def test_disabled_flag_uses_in_process(self):
        with mock.patch.object(mcp_client.settings, "flights_mcp_enabled", False):
            result = await self.client.status("KQ100", DATE)
        self.assertTrue(result["found"])
        # The MCP session should never have been started.
        self.assertIsNone(self.client._task)


if __name__ == "__main__":
    unittest.main()
