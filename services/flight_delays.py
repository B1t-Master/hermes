"""Mock flight-delay data: deterministic per-day, SQLite-backed, no network.

This is the single source of truth for the simulated flight schedule. The
`services/flights_mcp.py` MCP server exposes it as tools (`get_flight_status`,
`list_delays`), and the bot queries those tools through the MCP client
(`app/agents/mcp_client.py`) so it is a real MCP consumer. The in-process
functions here remain the fallback (and back the MCP server itself), so every
caller sees identical data.

Determinism contract: delays are generated from a RNG seeded by the calendar
date, so all lookups for a given day return stable answers, and the answers
change (plausibly) the next day. The SQLite file is just a cache of the
current day's roll; it is regenerated whenever the stored date differs.

No real flight data exists for hermes; every route and time here is invented
prototype filler.
"""

from __future__ import annotations

import os
import random
import re
import sqlite3
from datetime import date as date_cls

DEFAULT_DB_PATH = os.environ.get("FLIGHTS_DB_PATH", ".flights.sqlite")

# (flight_no, origin, destination, scheduled_departure, scheduled_arrival)
# Arrival times carry a "+1" suffix when they land the next calendar day.
SCHEDULE: list[tuple[str, str, str, str, str]] = [
    ("KQ100", "NBO", "LHR", "21:40", "05:15+1"),
    ("KQ101", "LHR", "NBO", "19:15", "06:10+1"),
    ("KQ102", "NBO", "CDG", "23:05", "06:20+1"),
    ("KQ112", "CDG", "NBO", "20:00", "06:30+1"),
    ("KQ116", "NBO", "FRA", "22:30", "06:00+1"),
    ("KQ117", "FRA", "NBO", "20:30", "06:45+1"),
    ("KQ306", "NBO", "DXB", "06:00", "11:30"),
    ("KQ305", "DXB", "NBO", "13:30", "18:15"),
    ("KQ310", "NBO", "JNB", "08:30", "12:40"),
    ("KQ312", "NBO", "CPT", "09:15", "13:50"),
    ("KQ400", "NBO", "DAR", "08:00", "10:20"),
    ("KQ444", "NBO", "MBA", "15:00", "16:30"),
    ("KQ418", "NBO", "EBB", "10:45", "12:30"),
    ("KQ720", "NBO", "CMB", "23:55", "08:40+1"),
    ("KQ868", "DOH", "NBO", "20:30", "03:40+1"),
    ("KQ870", "NBO", "DOH", "12:40", "18:45"),
    ("KQ1120", "NBO", "ACC", "09:30", "11:45"),
]

STATUS_ON_TIME = "ON_TIME"
STATUS_DELAYED = "DELAYED"
STATUS_CANCELLED = "CANCELLED"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS flights (
    flight_no           TEXT PRIMARY KEY,
    origin              TEXT NOT NULL,
    destination         TEXT NOT NULL,
    scheduled_departure TEXT NOT NULL,
    scheduled_arrival   TEXT NOT NULL,
    delay_minutes       INTEGER,
    status              TEXT NOT NULL,
    last_updated_for    TEXT NOT NULL
)
"""


def canonical_date(day: str | None = None) -> str:
    """Validate/normalize a user-supplied date; default to today."""
    if day is None:
        return date_cls.today().isoformat()
    candidate = day.strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
        raise ValueError(f"date must be YYYY-MM-DD, got {day!r}")
    date_cls.fromisoformat(candidate)  # raises on impossible dates
    return candidate


def _roll(rng: random.Random) -> tuple[str, int | None]:
    """One (status, delay_minutes) roll for a flight on a seeded day."""
    if rng.random() < 0.02:
        return STATUS_CANCELLED, None
    r = rng.random()
    if r < 0.70:
        return STATUS_ON_TIME, rng.randint(0, 14)
    if r < 0.92:
        return STATUS_DELAYED, rng.randint(15, 59)
    return STATUS_DELAYED, rng.randint(60, 240)


def _seed_day(conn: sqlite3.Connection, day: str) -> None:
    """Regenerate today's rolls when the cache is stale (or missing)."""
    stale = conn.execute("SELECT 1 FROM flights LIMIT 1").fetchone() is None
    if not stale:
        current = conn.execute("SELECT last_updated_for FROM flights LIMIT 1").fetchone()[0]
        stale = current != day
    if not stale:
        return
    rng = random.Random(f"hermes-flights:{day}")
    conn.execute("DELETE FROM flights")
    rows: list[tuple[str, str, str, str, str, int | None, str, str]] = []
    for flight_no, origin, destination, dep, arr in SCHEDULE:
        status, delay = _roll(rng)
        rows.append((flight_no, origin, destination, dep, arr, delay, status, day))
    conn.executemany(
        "INSERT INTO flights (flight_no, origin, destination, scheduled_departure,"
        " scheduled_arrival, delay_minutes, status, last_updated_for)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def _normalize_flight_no(flight_no: str) -> str:
    digits = re.sub(r"[^0-9]", "", flight_no.strip())
    return f"KQ{digits}" if digits else flight_no.strip().upper()


def _status_row(conn: sqlite3.Connection, flight_no: str, day: str) -> sqlite3.Row | None:
    _seed_day(conn, day)
    return conn.execute(
        "SELECT * FROM flights WHERE flight_no = ?", (_normalize_flight_no(flight_no),)
    ).fetchone()


def get_flight_status(flight_no: str, day: str | None = None, db_path: str | None = None) -> dict:
    """Status for one flight on a date (default: today).

    Returns a JSON-able dict with `found` always set so callers (the LLM, MCP
    clients) can branch on a clean shape instead of exceptions.
    """
    day = canonical_date(day)
    conn = _connect(db_path)
    try:
        row = _status_row(conn, flight_no, day)
        if row is None:
            return {"found": False, "flight_no": _normalize_flight_no(flight_no), "date": day}
        return {
            "found": True,
            "flight_no": row["flight_no"],
            "route": f"{row['origin']}-{row['destination']}",
            "origin": row["origin"],
            "destination": row["destination"],
            "scheduled_departure": row["scheduled_departure"],
            "scheduled_arrival": row["scheduled_arrival"],
            "delay_minutes": row["delay_minutes"],
            "status": row["status"],
            "date": day,
        }
    finally:
        conn.close()


def list_delays(
    route: str | None = None,
    day: str | None = None,
    min_delay: int = 15,
    db_path: str | None = None,
) -> list[dict]:
    """Flights delayed (>= `min_delay` minutes) on a date.

    `route` is an optional "NBO-LHR" pair or a single airport code ("NBO");
    when omitted every genuinely delayed flight is returned, worst first.
    """
    day = canonical_date(day)
    conn = _connect(db_path)
    try:
        _seed_day(conn, day)
        rows = conn.execute(
            "SELECT * FROM flights WHERE status = ? AND delay_minutes >= ?",
            (STATUS_DELAYED, min_delay),
        ).fetchall()
        flights = [
            {
                "flight_no": r["flight_no"],
                "route": f"{r['origin']}-{r['destination']}",
                "origin": r["origin"],
                "destination": r["destination"],
                "scheduled_departure": r["scheduled_departure"],
                "scheduled_arrival": r["scheduled_arrival"],
                "delay_minutes": r["delay_minutes"],
                "status": r["status"],
                "date": day,
            }
            for r in rows
        ]
        flights.sort(key=lambda f: f["delay_minutes"], reverse=True)
        if not route:
            return flights
        codes = {c.strip().upper() for c in re.split(r"\s*[-/]\s*|\s+", route.strip()) if c.strip()}
        if len(codes) == 1:
            code = codes.pop()
            return [f for f in flights if code in (f["origin"], f["destination"])]
        return [f for f in flights if f["origin"] in codes and f["destination"] in codes]
    finally:
        conn.close()