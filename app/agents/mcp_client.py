"""MCP client manager for multiple stdio MCP servers.

Supports:
- `hermes-flights` (tracking/status/delays) - existing
- `hermes-ops` (bookings/rebookings/tracking/operations) - new in Phase 4

Lifecycle: a single background task owns each stdio session (anyio task groups
must be entered and exited in the same task, so we cannot open it per request).
Requests await a ready event, then call tools on the shared session. If the
server cannot be reached, lookups fall back to the in-process library so a
request never takes the whole turn down; MCP can be toggled via config.
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
_FLIGHTS_SERVER_MODULE = "services.flights_mcp"
_OPS_SERVER_MODULE = "services.ops_mcp"


class MCPServerClient:
    """Persistent stdio MCP session against any MCP server."""

    def __init__(self, server_module: str = _FLIGHTS_SERVER_MODULE, task_name: str = "flights-mcp") -> None:
        self._server_module = server_module
        self._task_name = task_name
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
            self._task = asyncio.create_task(self._serve(), name=self._task_name)
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
            logger.warning("%s MCP shutdown raised", self._server_module, exc_info=True)
        self._task = None
        self._session = None

    # --- tools -----------------------------------------------------------
    async def call_tool(self, name: str, arguments: dict) -> object:
        """Call one MCP tool and return its decoded JSON payload."""
        await self.start()
        session = self._session
        if session is None:
            raise RuntimeError(f"{self._server_module} MCP session is not available")
        result = await session.call_tool(name, arguments)
        if result.isError:
            raise RuntimeError(f"{self._server_module} MCP tool {name} failed: {result.content}")
        if not result.content:
            return None
        # FastMCP may return JSON in text blocks
        content = result.content[0]
        text = getattr(content, "text", None)
        if text is None:
            # try model_dump/str
            if hasattr(content, "model_dump"):
                data = content.model_dump()
                if isinstance(data, dict) and "text" in data:
                    text = data["text"]
            if text is None:
                text = str(content)
        try:
            return json.loads(text)
        except Exception:
            return text

    async def list_tools(self):
        """Discover available tools from this server."""
        await self.start()
        session = self._session
        if session is None:
            raise RuntimeError(f"{self._server_module} MCP session is not available")
        return await session.list_tools()


class FlightsClient:
    """Flight status and delay helpers with MCP + fallback."""

    def __init__(self, client: MCPServerClient | None = None) -> None:
        if client is None:
            client = MCPServerClient(_FLIGHTS_SERVER_MODULE, "flights-mcp")
        self._client = client

    async def start(self) -> None:
        await self._client.start()

    async def status(self, flight_no: str, date: str | None = None) -> dict:
        """One flight's status, via MCP with an in-process fallback."""
        if settings.flights_mcp_enabled:
            arguments: dict = {"flight_no": flight_no}
            if date:
                arguments["date"] = date
            try:
                payload = await self._client.call_tool("get_flight_status", arguments)
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
                payload = await self._client.call_tool("list_delays", arguments)
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

    async def call_tool(self, name: str, arguments: dict) -> object:
        return await self._client.call_tool(name, arguments)

    async def list_tools(self):
        return await self._client.list_tools()

    async def stop(self) -> None:
        await self._client.stop()

    @property
    def _task(self):
        return self._client._task

    @property
    def _session(self):
        return self._client._session

    @property
    def _ready(self):
        return self._client._ready

    @property
    def _stop_event(self):
        return self._client._stop


# Keep the old name for backward compatibility
class FlightsMCPClient(FlightsClient):
    def __init__(self, server_module: str = _FLIGHTS_SERVER_MODULE, task_name: str = "flights-mcp") -> None:
        client = MCPServerClient(server_module, task_name)
        super().__init__(client)



# Process-wide MCP client singletons
_flights_client: MCPServerClient | None = None
_ops_client: MCPServerClient | None = None
_flights_helper: FlightsClient | None = None


def get_flights_client() -> FlightsClient:
    """Process-wide MCP client singleton for flights."""
    global _flights_client, _flights_helper
    if _flights_client is None:
        _flights_client = MCPServerClient(_FLIGHTS_SERVER_MODULE, "flights-mcp")
    if _flights_helper is None:
        _flights_helper = FlightsClient(_flights_client)
    return _flights_helper


def get_ops_client() -> MCPServerClient:
    """Process-wide MCP client singleton for ops (bookings/rebookings/tracking)."""
    global _ops_client
    if _ops_client is None:
        _ops_client = MCPServerClient(_OPS_SERVER_MODULE, "ops-mcp")
    return _ops_client


async def shutdown_all() -> None:
    """Stop all MCP clients gracefully."""
    global _flights_client, _ops_client, _flights_helper
    if _flights_client is not None:
        await _flights_client.stop()
    if _ops_client is not None:
        await _ops_client.stop()
    _flights_client = None
    _ops_client = None
    _flights_helper = None

