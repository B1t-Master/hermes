"""MCP client for the mock KQ flight service.

Flight status and delay questions are answered by the `hermes-flights` MCP
server (`services/flights_mcp.py`) over stdio, the same tools any external MCP
client uses, instead of importing `services.flight_delays` directly. This is
what makes the bot a real MCP consumer rather than a second in-process caller.

Lifecycle: a single background task owns the stdio session (anyio task groups
must be entered and exited in the same task, so we cannot open it per request).
Requests await a ready event, then call tools on the shared session. If the
server cannot be reached, lookups fall back to the in-process library so a
flight question never takes the whole turn down; `FLIGHTS_MCP_ENABLED=false`
skips MCP entirely.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import settings
from services import flight_delays

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_MODULE = "services.flights_mcp"


class FlightsMCPClient:
    """Persistent stdio MCP session against the mock flight server."""

    def __init__(self, server_module: str = _SERVER_MODULE) -> None:
        self._server_module = server_module
        self._task: asyncio.Task[None] | None = None
        self._session: ClientSession | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._error: BaseException | None = None
        self._start_lock = asyncio.Lock()

    # --- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        """Spawn the server and wait until the session is initialized."""
        async with self._start_lock:
            if self._task is not None and not self._task.done():
                return
            self._ready = asyncio.Event()
            self._stop = asyncio.Event()
            self._error = None
            self._task = asyncio.create_task(self._serve(), name="flights-mcp")
        await self._ready.wait()
        if self._error is not None:
            raise self._error

    async def _serve(self) -> None:
        """Own the stdio session for as long as the client is running."""
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", self._server_module],
            cwd=str(_REPO_ROOT),
            env=dict(os.environ),
        )
        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._session = session
                self._ready.set()
                logger.info("Connected to the %s MCP server", self._server_module)
                await self._stop.wait()
        except BaseException as exc:  # surfaced to callers via start()/call_tool()
            self._error = exc
            self._ready.set()
        finally:
            self._session = None

    async def stop(self) -> None:
        """Signal the session task to close and wait for it to finish."""
        task = self._task
        if task is None:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(task, timeout=5)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        except Exception:
            logger.warning("Flights MCP shutdown raised", exc_info=True)
        self._task = None
        self._session = None

    # --- tools -----------------------------------------------------------
    async def call_tool(self, name: str, arguments: dict) -> object:
        """Call one MCP tool and return its decoded JSON payload."""
        await self.start()
        session = self._session
        if session is None:
            raise RuntimeError("flights MCP session is not available")
        result = await session.call_tool(name, arguments)
        if result.isError:
            raise RuntimeError(f"flights MCP tool {name} failed: {result.content}")
        if not result.content:
            return None
        return json.loads(result.content[0].text)

    async def status(self, flight_no: str, date: str | None = None) -> dict:
        """One flight's status, via MCP with an in-process fallback."""
        if settings.flights_mcp_enabled:
            arguments: dict = {"flight_no": flight_no}
            if date:
                arguments["date"] = date
            try:
                payload = await self.call_tool("get_flight_status", arguments)
                if isinstance(payload, dict) and payload:
                    return payload
            except Exception:
                logger.warning(
                    "get_flight_status over MCP failed; using in-process service",
                    exc_info=True,
                )
        return await asyncio.to_thread(flight_delays.get_flight_status, flight_no, date)

    async def delays(self, route: str | None = None, date: str | None = None) -> list[dict]:
        """Delayed flights, via MCP with an in-process fallback."""
        if settings.flights_mcp_enabled:
            arguments: dict = {}
            if route:
                arguments["route"] = route
            if date:
                arguments["date"] = date
            try:
                payload = await self.call_tool("list_delays", arguments)
                if isinstance(payload, dict) and isinstance(payload.get("delays"), list):
                    return payload["delays"]
                if isinstance(payload, list):
                    return payload
            except Exception:
                logger.warning(
                    "list_delays over MCP failed; using in-process service",
                    exc_info=True,
                )
        return await asyncio.to_thread(flight_delays.list_delays, route, date)


_client: FlightsMCPClient | None = None


def get_flights_client() -> FlightsMCPClient:
    """Process-wide MCP client singleton."""
    global _client
    if _client is None:
        _client = FlightsMCPClient()
    return _client
