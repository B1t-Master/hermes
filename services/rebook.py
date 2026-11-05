"""Mock rebooking service: deterministic, no network, no real reservation system.

Simplified for the demo: once the passenger supplies their booking reference and
a new travel date, we pick an available flight from the SQLite schedule
(`.flights.sqlite`), move the passenger onto it for that date, and record the
rebooking in the same database. The flight *number* is not asked of the
passenger — availability is chosen for them.

Every value here is invented prototype filler.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import sqlite3
from datetime import date as date_cls
from datetime import timedelta

from services import flight_delays

DEFAULT_DB_PATH = os.environ.get("FLIGHTS_DB_PATH", ".flights.sqlite")

_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

# Trailing/embedded time-of-day words that do not change the calendar date.
_TIME_WORDS = re.compile(
    r"\b(morning|afternoon|evening|night|tonight|noon|midday|pm|am)\b", re.IGNORECASE
)

_RELATIVE_DAY_WORDS = (
    (re.compile(r"^(?:the\s+)?day after tomorrow$", re.IGNORECASE), 2),
    (re.compile(r"^(?:tomorrow|next day)$", re.IGNORECASE), 1),
    (re.compile(r"^today$", re.IGNORECASE), 0),
    (re.compile(r"^next week$", re.IGNORECASE), 7),
    (re.compile(r"^in (\d{1,2}) days?$", re.IGNORECASE), None),
)

# "Move me onto the next available flight" - the passenger has not chosen a
# date, and a disrupted flight is rebooked automatically onto the next
# available departure rather than being held up for a date.
_NEXT_AVAILABLE = re.compile(
    r"^(?:next available|next available flight|as soon as possible|"
    r"earliest available|earliest available flight|any available|"
    r"the next one|next one)$",
    re.IGNORECASE,
)


def _parse_new_date(value: str, base: date_cls) -> date_cls | None:
    """Understand 'tomorrow', 'tomorrow afternoon', 'next Friday', ISO dates."""
    if not value:
        return None
    text = value.strip().lower()
    # Drop time-of-day qualifiers: "tomorrow afternoon" -> "tomorrow".
    text = _TIME_WORDS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.")

    if _NEXT_AVAILABLE.match(text):
        # Next available departure: the first day after today.
        return base + timedelta(days=1)

    for pattern, days in _RELATIVE_DAY_WORDS:
        match = pattern.match(text)
        if match:
            offset = int(match.group(1)) if days is None else days
            return base + timedelta(days=offset)

    match = re.match(r"^next\s+(" + "|".join(_WEEKDAYS) + r")$", text)
    if match:
        target = _WEEKDAYS.index(match.group(1))
        ahead = (target - base.weekday()) % 7 or 7
        return base + timedelta(days=ahead)

    match = re.match(r"^(" + "|".join(_WEEKDAYS) + r")$", text)
    if match:
        target = _WEEKDAYS.index(match.group(1))
        ahead = (target - base.weekday()) % 7
        return base + timedelta(days=ahead)

    try:
        return date_cls.fromisoformat(flight_delays.canonical_date(text))
    except ValueError:
        return None


def _reference(
    flight_no: str, new_date: str, ticket_number: str, passenger_name: str | None
) -> str:
    seed = f"hermes-rebook:{flight_no}:{new_date}:{ticket_number}:{passenger_name or 'guest'}"
    return "HRM-" + hashlib.sha256(seed.encode()).hexdigest()[:6].upper()


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _schedule_flights(conn: sqlite3.Connection) -> list[dict]:
    """Every flight in the schedule, seeding the cache if it is empty."""
    try:
        rows = conn.execute(
            "SELECT flight_no, origin, destination, scheduled_departure, "
            "scheduled_arrival, status FROM flights"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    if not rows:
        # No cache yet: fall back to the static schedule so a rebooking can
        # still pick a flight before any status lookup has run.
        return [
            {
                "flight_no": flight_no,
                "origin": origin,
                "destination": destination,
                "scheduled_departure": dep,
                "scheduled_arrival": arr,
                "status": "ON_TIME",
            }
            for flight_no, origin, destination, dep, arr in flight_delays.SCHEDULE
        ]
    return [dict(row) for row in rows]


def _record_rebooking(
    conn: sqlite3.Connection,
    ticket_number: str,
    flight_no: str,
    new_date: str,
    reference: str,
) -> None:
    """Persist the new flight assignment in the same SQLite database."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rebookings ("
        " ticket_number TEXT, flight_no TEXT, new_date TEXT,"
        " reference TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,"
        " PRIMARY KEY (ticket_number, new_date))"
    )
    conn.execute(
        "INSERT OR REPLACE INTO rebookings"
        " (ticket_number, flight_no, new_date, reference)"
        " VALUES (?, ?, ?, ?)",
        (ticket_number, flight_no, new_date, reference),
    )
    conn.commit()


def rebook_flight(
    flight_no: str | None = None,
    ticket_number: str | None = None,
    new_date: str | None = None,
    passenger_name: str | None = None,
    reason: str | None = None,
    *,
    today: date_cls | None = None,
    rng: random.Random | None = None,
    db_path: str | None = None,
) -> dict:
    """Rebook a passenger onto an available flight.

    Requires the booking reference / e-ticket number and the new travel date.
    When no flight number is supplied, returns the available alternatives so
    the passenger can choose one. When a flight number is supplied, confirms
    the rebooking (the caller persists it to Postgres via ``confirm_rebook``).
    """
    if not ticket_number:
        return {
            "rebooked": False,
            "needs_ticket": True,
            "reason": (
                "A booking reference or e-ticket number is required before "
                "rebooking. Ask the passenger for it."
            ),
        }

    base = today or date_cls.today()
    if not new_date:
        return {
            "rebooked": False,
            "needs_date": True,
            "reason": (
                "The passenger's new travel date is required before rebooking. "
                "Ask for it (for example, 'tomorrow' or a calendar date)."
            ),
        }

    target = _parse_new_date(new_date, base)
    if target is None:
        return {
            "rebooked": False,
            "date_invalid": True,
            "reason": (
                f"Could not understand the date {new_date!r}. Ask the passenger "
                "to rephrase, for example 'tomorrow' or '2025-04-20'."
            ),
        }
    target_iso = target.isoformat()

    # Available alternatives come from the predetermined demo flight list.
    from services import booking as _booking

    alternatives = [
        {
            "flight_no": f["flight_no"],
            "origin": f["origin"],
            "destination": f["destination"],
            "route": f"{f['origin']}-{f['destination']}",
            "date": target_iso,
            "depart_time": f["depart_time"],
            "arrival_time": f["arrival_time"],
            "fare": _booking.DEMO_FARE,
            "currency": _booking.DEMO_CURRENCY,
        }
        for f in _booking.AVAILABLE_FLIGHTS
    ]

    reference = _reference(flight_no, target_iso, ticket_number, passenger_name)
    chosen = next((f for f in alternatives if f["flight_no"] == flight_no), alternatives[0])

    # Persist the rebooking in the mock SQLite schedule database.
    conn = _connect(db_path)
    try:
        _record_rebooking(conn, ticket_number, chosen["flight_no"], target_iso, reference)
    finally:
        conn.close()

    return {
        "rebooked": True,
        "reference": reference,
        "confirmation_reference": reference,
        "ticket_number": ticket_number,
        "original_flight_no": flight_no,
        "flight_no": chosen["flight_no"],
        "route": chosen["route"],
        "origin": chosen["origin"],
        "destination": chosen["destination"],
        "original_date": base.isoformat(),
        "date": target_iso,
        "scheduled_departure": chosen["depart_time"],
        "scheduled_arrival": chosen["arrival_time"],
        "status": "ON_TIME",
        "passenger_name": passenger_name,
        "reason": reason,
        "message": (
            f"Rebooked to {chosen['flight_no']} ({chosen['origin']}-"
            f"{chosen['destination']}) on {target_iso}. "
            f"Confirmation reference {reference}."
        ),
    }


def confirmation_email(result: dict) -> tuple[str, str]:
    """Subject and body for the rebooking confirmation email."""
    subject = f"Your Kenya Airways rebooking confirmation {result['reference']}"
    passenger = result.get("passenger_name") or "passenger"
    body = (
        f"Dear {passenger},\n\n"
        f"Your flight has been rebooked.\n\n"
        f"New flight details\n"
        f"  Flight:                 {result['flight_no']} ({result['route']})\n"
        f"  Date:                   {result['date']}\n"
        f"  Scheduled departure:    {result['scheduled_departure']}\n"
        f"  Scheduled arrival:      {result['scheduled_arrival']}\n"
        f"  Status:                 {result['status']}\n"
        f"  Booking reference:      {result['ticket_number']}\n"
        f"  Confirmation reference: {result['reference']}\n\n"
        f"Thank you for flying Kenya Airways.\n"
    )
    return subject, body
