"""Flight search, booking, fare, and e-ticket service (prototype).

The schedule is the same mock data the flight-status service uses
(``services/flight_delays.SCHEDULE``); fares and availability are derived
deterministically from the flight, route and travel date so a search for the
same day always returns the same quote.

Database writes go through the shared async session (PostgreSQL). The ops MCP
server imports this module too, so the LLM and the API see one implementation.
"""

from __future__ import annotations

import asyncio
import calendar
import os
import sqlite3
import base64
import hashlib
import logging
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from datetime import date as date_cls

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Booking, BookingStatus, Payment, PaymentStatus
from services import email_real, flight_delays, mpesa, rebook

logger = logging.getLogger(__name__)

CURRENCY = "KES"

# Airport code -> searchable names. Every code in the mock schedule is listed.
AIRPORTS: dict[str, dict[str, str]] = {
    "NBO": {"city": "Nairobi", "name": "Jomo Kenyatta International", "country": "Kenya"},
    "LHR": {"city": "London", "name": "Heathrow", "country": "United Kingdom"},
    "CDG": {"city": "Paris", "name": "Charles de Gaulle", "country": "France"},
    "FRA": {"city": "Frankfurt", "name": "Frankfurt am Main", "country": "Germany"},
    "DXB": {"city": "Dubai", "name": "Dubai International", "country": "UAE"},
    "JNB": {"city": "Johannesburg", "name": "O. R. Tambo", "country": "South Africa"},
    "CPT": {"city": "Cape Town", "name": "Cape Town International", "country": "South Africa"},
    "DAR": {"city": "Dar es Salaam", "name": "Julius Nyerere", "country": "Tanzania"},
    "MBA": {"city": "Mombasa", "name": "Moi International", "country": "Kenya"},
    "EBB": {"city": "Entebbe", "name": "Entebbe International", "country": "Uganda"},
    "CMB": {"city": "Colombo", "name": "Bandaranaike", "country": "Sri Lanka"},
    "DOH": {"city": "Doha", "name": "Hamad International", "country": "Qatar"},
    "ACC": {"city": "Accra", "name": "Kotoka International", "country": "Ghana"},
}

# Extra names/aliases passengers use in chat.
_ALIASES: dict[str, str] = {
    "nairobi": "NBO",
    "jomo kenyatta": "NBO",
    "jka": "NBO",
    "london": "LHR",
    "heathrow": "LHR",
    "paris": "CDG",
    "charles de gaulle": "CDG",
    "frankfurt": "FRA",
    "dubai": "DXB",
    "johannesburg": "JNB",
    "joburg": "JNB",
    "jo'burg": "JNB",
    "cape town": "CPT",
    "capetown": "CPT",
    "dar es salaam": "DAR",
    "dar": "DAR",
    "mombasa": "MBA",
    "entebbe": "EBB",
    "colombo": "CMB",
    "doha": "DOH",
    "accra": "ACC",
}
for _code, _info in AIRPORTS.items():
    _ALIASES.setdefault(_code.lower(), _code)
    _ALIASES.setdefault(_info["city"].lower(), _code)
    _ALIASES.setdefault(_info["name"].lower(), _code)

_MONTHS = {
    name.lower(): index
    for index, name in enumerate(calendar.month_name)
    if name
}
_MONTHS.update(
    {
        name.lower(): index
        for index, name in enumerate(calendar.month_abbr)
        if name
    }
)

# --- Demo available-flights list -------------------------------------------
# Predetermined schedule for the booking demo. Every flight is one-way and the
# fare is a flat 1 KES so the demo payment path is trivial. A passenger can
# hold more than one ticket (bookings.passenger_id is a 1:N foreign key).
AVAILABLE_FLIGHTS: list[dict[str, str]] = [
    {
        "flight_no": "KQ100",
        "origin": "NBO",
        "destination": "LHR",
        "depart_time": "21:40",
        "arrival_time": "05:15+1",
    },
    {
        "flight_no": "KQ306",
        "origin": "NBO",
        "destination": "DXB",
        "depart_time": "06:00",
        "arrival_time": "11:30",
    },
    {
        "flight_no": "KQ444",
        "origin": "NBO",
        "destination": "MBA",
        "depart_time": "15:00",
        "arrival_time": "16:30",
    },
    {
        "flight_no": "KQ310",
        "origin": "NBO",
        "destination": "JNB",
        "depart_time": "08:30",
        "arrival_time": "12:40",
    },
    {
        "flight_no": "KQ870",
        "origin": "NBO",
        "destination": "DOH",
        "depart_time": "12:40",
        "arrival_time": "18:45",
    },
]

DEMO_FARE = 1.00  # flat demo fare in KES
DEMO_CURRENCY = "KES"


def _demo_flights_for_route(origin_code: str, destination_code: str) -> list[dict]:
    """Return the demo flights matching a route (origin, destination)."""
    return [
        f
        for f in AVAILABLE_FLIGHTS
        if f["origin"] == origin_code and f["destination"] == destination_code
    ]


def _demo_fare() -> float:
    return DEMO_FARE


def _normalise_place(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", value.lower()).strip()


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u2013", "-").replace("\u2014", "-")).strip()


def resolve_airport(value: str | None) -> str | None:
    """Resolve a city/airport name or IATA code to a code, or ``None``."""
    if not value:
        return None
    text = _normalise_place(value)
    if not text:
        return None
    if text.upper() in AIRPORTS:
        return text.upper()
    if text in _ALIASES:
        return _ALIASES[text]
    # A city embedded in a longer phrase ("Nairobi, Kenya").
    for alias, code in sorted(_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return code
    return None


def airport_label(code: str | None) -> str:
    """``NBO`` -> ``Nairobi (NBO)`` for answers and cards."""
    if not code:
        return ""
    info = AIRPORTS.get(code.upper())
    if not info:
        return code.upper()
    return f"{info['city']} ({code.upper()})"


def parse_travel_date(value: str | None, *, today: date_cls | None = None) -> date_cls | None:
    """Understand ISO dates, ``12 December`` and relative words.

    Returns ``None`` when the text cannot be understood; callers ask the
    passenger for a date rather than inventing one.
    """
    if not value:
        return None
    text = _normalise_text(value).strip(" .,")
    base = today or date_cls.today()

    for pattern in (r"\b(\d{4})-(\d{2})-(\d{2})\b", r"\b(\d{4})/(\d{1,2})/(\d{1,2})\b"):
        match = re.search(pattern, text)
        if match:
            try:
                return date_cls(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                return None
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if match:
        try:
            return date_cls(int(match.group(3)), int(match.group(2)), int(match.group(1)))
        except ValueError:
            return None

    # "12 December 2026", "12 Dec", "December 12", "Dec 12 2026".
    month_names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    match = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_names})\.?(?:\s+(\d{{4}}))?\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        match = re.search(
            rf"\b({month_names})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+(\d{{4}}))?\b",
            text,
            re.IGNORECASE,
        )
        if match:
            month, day, year = match.group(1), match.group(2), match.group(3)
        else:
            month = day = year = None
    else:
        day, month, year = match.group(1), match.group(2), match.group(3)
    if month and day:
        try:
            candidate = date_cls(
                int(year) if year else base.year, _MONTHS[month.lower()], int(day))
        except (KeyError, ValueError):
            return None
        if not year and candidate < base:
            candidate = candidate.replace(year=candidate.year + 1)
        return candidate

    parsed = rebook._parse_new_date(text, base)
    return parsed


def _place_regex() -> str:
    aliases = "|".join(
        re.escape(alias) for alias in sorted(_ALIASES, key=len, reverse=True)
    )
    return rf"(?:{aliases})"


_PLACES = _place_regex()
_FROM_TO = re.compile(
    r"\bfrom\s+(?P<origin>.+?)\s+to\s+(?P<destination>.+?)"
    r"(?=\s+(?:on|for|departing|returning|next|tomorrow|today|this|in)\b|[,.!?;]|$)",
    re.IGNORECASE,
)
_TO_FROM = re.compile(
    r"\b(?:fly|travel|go)\s+to\s+(?P<destination>.+?)\s+from\s+(?P<origin>.+?)"
    r"(?=\s+(?:on|for|next|tomorrow|today|this|in)\b|[,.!?;]|$)",
    re.IGNORECASE,
)
_PAIR = re.compile(
    rf"\b(?P<origin>{_PLACES})\s+(?:to|->|-)\s+(?P<destination>{_PLACES})\b",
    re.IGNORECASE,
)


def parse_route(text: str) -> tuple[str, str] | None:
    """Extract a (origin, destination) airport pair from free text."""
    if not text:
        return None
    normalised = _normalise_text(text)

    for pattern in (_FROM_TO, _TO_FROM):
        match = pattern.search(normalised)
        if match:
            origin = resolve_airport(match.group("origin"))
            destination = resolve_airport(match.group("destination"))
            if origin and destination and origin != destination:
                return origin, destination

    match = _PAIR.search(normalised)
    if match:
        origin = resolve_airport(match.group("origin"))
        destination = resolve_airport(match.group("destination"))
        if origin and destination and origin != destination:
            return origin, destination

    # Last resort: two known places mentioned in order ("Nairobi London").
    found = [
        resolve_airport(match.group(0))
        for match in re.finditer(_PLACES, normalised, re.IGNORECASE)
    ]
    found = [code for code in found if code]
    # Return two DISTINCT airports, not the same code appearing twice.
    distinct: list[str] = []
    for code in found:
        if code not in distinct:
            distinct.append(code)
        if len(distinct) == 2:
            return distinct[0], distinct[1]
    return None


def _fare(flight_no: str, origin: str, destination: str, travel_date: date_cls) -> int:
    """Flat demo fare: 1 KES for every flight."""
    return int(DEMO_FARE)


def _seats_left(flight_no: str, travel_date: date_cls) -> int:
    """Base capacity (deterministic) minus real bookings already taken."""
    digest = int(
        hashlib.sha256(
            f"seats:{flight_no}:{travel_date.isoformat()}".encode()).hexdigest()[:6],
        16,
    )
    base = 1 + digest % 9
    return max(0, base - _seats_taken(flight_no, travel_date))


def _seats_db_path() -> str:
    return os.environ.get("FLIGHTS_DB_PATH", ".flights.sqlite")


def _ensure_seat_inventory() -> None:
    """Create the seat-inventory table in the flights SQLite DB if missing."""
    try:
        conn = sqlite3.connect(_seats_db_path())
        conn.execute(
            "CREATE TABLE IF NOT EXISTS seat_inventory ("
            "flight_no TEXT NOT NULL, travel_date TEXT NOT NULL, "
            "seats_taken INTEGER NOT NULL DEFAULT 0, "
            "PRIMARY KEY (flight_no, travel_date))"
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("Could not ensure seat_inventory table")


def _seats_taken(flight_no: str, travel_date: date_cls) -> int:
    _ensure_seat_inventory()
    try:
        conn = sqlite3.connect(_seats_db_path())
        row = conn.execute(
            "SELECT seats_taken FROM seat_inventory "
            "WHERE flight_no = ? AND travel_date = ?",
            (flight_no, travel_date.isoformat()),
        ).fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception:
        logger.exception("Could not read seat inventory")
        return 0


async def decrement_seats(flight_no: str, travel_date: str | date_cls | None) -> None:
    """Decrement (by one) the available seats for a flight on a date."""
    if not flight_no or not travel_date:
        return
    if isinstance(travel_date, str):
        try:
            parsed = date_cls.fromisoformat(travel_date)
        except ValueError:
            return
    else:
        parsed = travel_date
    await asyncio.to_thread(_decrement_seats_sync, flight_no, parsed)


def _decrement_seats_sync(flight_no: str, travel_date: date_cls) -> None:
    _ensure_seat_inventory()
    try:
        conn = sqlite3.connect(_seats_db_path())
        conn.execute(
            "INSERT INTO seat_inventory (flight_no, travel_date, seats_taken) "
            "VALUES (?, ?, 1) "
            "ON CONFLICT(flight_no, travel_date) "
            "DO UPDATE SET seats_taken = seats_taken + 1",
            (flight_no, travel_date.isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("Could not decrement seats for %s %s",
                         flight_no, travel_date)


def _schedule_row(flight_no: str) -> tuple[str, str, str, str, str] | None:
    wanted = re.sub(r"\s+", "", (flight_no or "").upper())
    for row in flight_delays.SCHEDULE:
        if row[0].upper() == wanted:
            return row
    return None


def search_flights(
    origin: str,
    destination: str,
    travel_date: str | None,
    *,
    today: date_cls | None = None,
) -> dict:
    """Search the mock schedule for one route and date.

    Always returns a JSON-able dict with ``available`` set, so the LLM can
    branch on the shape instead of handling exceptions.
    """
    origin_code = resolve_airport(origin)
    destination_code = resolve_airport(destination)
    base = today or date_cls.today()
    if not origin_code or not destination_code:
        return {
            "available": False,
            "error": "I couldn't recognise that route. Use a city or airport code, "
            "for example Nairobi (NBO) to Mombasa (MBA).",
        }
    if origin_code == destination_code:
        return {"available": False, "error": "Origin and destination must be different."}
    parsed_date = parse_travel_date(travel_date, today=base)
    if parsed_date is None:
        return {
            "available": False,
            "needs_date": True,
            "error": "Which date would you like to travel? For example '12 December' "
            "or '2026-12-12'.",
        }
    if parsed_date < base:
        return {"available": False, "error": "That date is in the past. Choose a future date."}

    # Prefer the predetermined demo flight list (flat 1 KES fares); fall back
    # to the mock schedule for any other route.
    demo = _demo_flights_for_route(origin_code, destination_code)
    if demo:
        flights = [
            {
                "flight_no": f["flight_no"],
                "origin": f["origin"],
                "destination": f["destination"],
                "route": f"{f['origin']}-{f['destination']}",
                "date": parsed_date.isoformat(),
                "depart_time": f["depart_time"],
                "arrival_time": f["arrival_time"],
                "fare": _fare(f["flight_no"], f["origin"], f["destination"], parsed_date),
                "currency": CURRENCY,
                "seats_left": 9,
                "cabin": "Economy",
            }
            for f in demo
        ]
    else:
        flights = []
        for flight_no, dep_origin, dep_destination, depart, arrive in flight_delays.SCHEDULE:
            if dep_origin != origin_code or dep_destination != destination_code:
                continue
            flights.append(
                {
                    "flight_no": flight_no,
                    "origin": dep_origin,
                    "destination": dep_destination,
                    "route": f"{dep_origin}-{dep_destination}",
                    "date": parsed_date.isoformat(),
                    "depart_time": depart,
                    "arrival_time": arrive,
                    "fare": _fare(flight_no, dep_origin, dep_destination, parsed_date),
                    "currency": CURRENCY,
                    "seats_left": _seats_left(flight_no, parsed_date),
                    "cabin": "Economy",
                }
            )
    flights.sort(key=lambda item: item["depart_time"])
    if not flights:
        return {
            "available": False,
            "origin": origin_code,
            "destination": destination_code,
            "date": parsed_date.isoformat(),
            "error": (
                f"No Kenya Airways flights from {airport_label(origin_code)} to "
                f"{airport_label(destination_code)} on {parsed_date.isoformat()}. "
                "Try another date or route."
            ),
        }
    return {
        "available": True,
        "origin": origin_code,
        "destination": destination_code,
        "route": f"{origin_code}-{destination_code}",
        "date": parsed_date.isoformat(),
        "currency": CURRENCY,
        "flights": flights,
    }


def _new_pnr() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "KQ" + "".join(secrets.choice(alphabet) for _ in range(5))


async def _unique_pnr(db) -> str:
    for _ in range(20):
        candidate = _new_pnr()
        existing = (
            await db.execute(select(Booking.id).where(Booking.pnr == candidate))
        ).scalar_one_or_none()
        if existing is None:
            return candidate
    raise RuntimeError("Could not allocate a booking reference")


def booking_payload(booking: Booking) -> dict:
    """Prompt/card-friendly view of a booking."""
    return {
        "pnr": booking.pnr,
        "flight_no": booking.flight_no,
        "origin": booking.origin,
        "destination": booking.destination,
        "route": f"{booking.origin}-{booking.destination}",
        "date": booking.depart_date,
        "depart_time": booking.depart_time,
        "status": booking.status.value if booking.status else None,
        "amount": booking.amount,
        "currency": booking.currency,
        "passenger_name": booking.passenger_name,
        "passenger_email": booking.passenger_email,
        "eticket_number": booking.eticket_number,
    }


async def create_booking(
    flight_no: str,
    travel_date: str | None,
    *,
    passenger_name: str | None = None,
    passenger_email: str | None = None,
    passenger_phone: str | None = None,
    passenger_id: str | uuid.UUID | None = None,
    conversation_id: str | uuid.UUID | None = None,
    today: date_cls | None = None,
) -> dict:
    """Create a held booking and a pending M-Pesa payment record.

    The in-chat payment card collects the M-Pesa number and starts the STK
    push through :func:`app.agents.payments.start_payment`; this function never
    contacts M-Pesa. That keeps the card visible before the prompt and ensures
    every push uses the background poller and callback path.

    ``passenger_email`` is optional and is stored for later ticket delivery.
    ``passenger_phone`` is also optional; when supplied it is stored on the
    payment row, but the card remains the place where the passenger confirms
    the number and triggers the prompt.
    """
    row = _schedule_row(flight_no)
    if row is None:
        return {"booked": False, "error": f"Unknown flight {flight_no!r}."}
    flight_no, origin, destination, depart, _arrive = row
    base = today or date_cls.today()
    parsed_date = parse_travel_date(travel_date, today=base)
    if parsed_date is None:
        return {"booked": False, "needs_date": True, "error": "A travel date is required."}
    if parsed_date < base:
        return {"booked": False, "error": "That travel date is in the past."}

    amount = _fare(flight_no, origin, destination, parsed_date)
    passenger_uuid = None
    if passenger_id:
        try:
            passenger_uuid = uuid.UUID(str(passenger_id))
        except (ValueError, TypeError):
            passenger_uuid = None
    conversation_uuid = None
    if conversation_id:
        try:
            conversation_uuid = uuid.UUID(str(conversation_id))
        except (ValueError, TypeError):
            conversation_uuid = None

    async with SessionLocal() as db:
        pnr = await _unique_pnr(db)
        booking = Booking(
            pnr=pnr,
            passenger_id=passenger_uuid,
            passenger_email=passenger_email,
            passenger_name=passenger_name,
            flight_no=flight_no,
            origin=origin,
            destination=destination,
            depart_date=parsed_date.isoformat(),
            depart_time=depart,
            status=BookingStatus.HELD,
            amount=amount,
            currency=CURRENCY,
        )
        db.add(booking)
        await db.flush()
        payment = Payment(
            conversation_id=conversation_uuid,
            pnr=pnr,
            amount=amount,
            currency=CURRENCY,
            status=PaymentStatus.PENDING,
            phone=mpesa.normalize_phone(
                passenger_phone) if passenger_phone else None,
        )
        db.add(payment)
        await db.commit()
        await db.refresh(booking)
        await db.refresh(payment)

    payload = booking_payload(booking)
    payload.update(
        {
            "booked": True,
            "booking_id": str(booking.id),
            "payment_id": str(payment.id),
            "payment_status": payment.status.value,
            "pnr": pnr,
            "paid": False,
            "awaiting_phone": True,
            "message": (
                f"I've held {flight_no} ({origin}-{destination}) on "
                f"{parsed_date.isoformat()} at {depart} for {CURRENCY} {amount:,.0f}. "
                f"Booking reference: {pnr}.\n"
                "Enter your M-Pesa number in the payment card to receive the "
                "prompt."
            ),
        }
    )
    return payload


async def _issue_and_email_ticket(
    pnr: str, email: str, payload: dict
) -> dict:
    """Issue the e-ticket for a paid booking and email it to the passenger."""
    result = await issue_ticket(pnr, email=email)
    if result.get("ticketed"):
        payload.update(
            {
                "ticketed": True,
                "eticket_number": result.get("eticket_number"),
                "email_sent": result.get("email_sent", False),
                "email_to": result.get("email_to"),
                "message": (
                    f"Your e-ticket is confirmed and has been emailed to {email}. "
                    f"Booking reference: {pnr}. E-ticket number: "
                    f"{result.get('eticket_number')}."
                ),
            }
        )
    else:
        payload.update(
            {
                "ticketed": False,
                "message": (
                    f"Payment received for {pnr}, but I couldn't issue the "
                    f"e-ticket: {result.get('error')}. "
                    "Please contact customer care."
                ),
            }
        )
    return payload


async def complete_booking_email(
    pnr: str, email: str, *, passenger_name: str | None = None
) -> dict:
    """Add a passenger email to a paid booking and issue/email the e-ticket.

    Used after payment succeeds without an email on file: the passenger
    supplies their address, the ticket is issued and sent.
    """
    async with SessionLocal() as db:
        booking = (
            await db.execute(select(Booking).where(Booking.pnr == pnr.upper()))
        ).scalar_one_or_none()
        if booking is None:
            return {"ticketed": False, "error": f"Booking {pnr} was not found."}
        booking.passenger_email = email
        if passenger_name:
            booking.passenger_name = passenger_name
        await db.commit()
        await db.refresh(booking)
    return await _issue_and_email_ticket(pnr, email, booking_payload(booking))


async def set_booking_email(pnr: str, email: str) -> dict:
    """Store an optional e-ticket email without issuing the ticket.

    The payment card asks for this after payment succeeds. Ticketing stays with
    the agent dashboard; if the ticket was already issued before the email was
    supplied, send the existing e-ticket as a copy so the address is still
    useful.
    """
    normalized = (email or "").strip().lower()
    async with SessionLocal() as db:
        booking = (
            await db.execute(
                select(Booking).where(Booking.pnr == (pnr or "").upper())
            )
        ).scalar_one_or_none()
        if booking is None:
            return {"updated": False, "error": f"Booking {pnr} was not found."}
        booking.passenger_email = normalized
        await db.commit()
        await db.refresh(booking)
        payload = booking_payload(booking)
        already_ticketed = (
            booking.status == BookingStatus.TICKETED and bool(
                booking.eticket_number)
        )

    if not already_ticketed:
        return {"updated": True, **payload}

    subject, body = ticket_email(booking)
    html = ticket_email_html(booking)
    try:
        record = await asyncio.to_thread(
            email_real.send_email,
            normalized,
            subject,
            body,
            sender=settings_sender(),
            api_key=settings_key(),
            outbox_dir=settings_outbox(),
            html=html,
        )
    except Exception:
        logger.exception("E-ticket copy email failed for %s", pnr)
        payload.update(
            {
                "email_sent": False,
                "email_note": "The e-ticket copy could not be emailed.",
            }
        )
    else:
        payload.update(
            {
                "email_sent": True,
                "email_to": normalized,
                "email_provider": record.get("provider"),
            }
        )
    return {"updated": True, **payload}


async def confirm_rebook(
    ticket_number: str,
    new_date: str,
    flight_no: str,
    *,
    email: str | None = None,
    passenger_name: str | None = None,
    today: date_cls | None = None,
) -> dict:
    """Confirm a rebooking: update the existing booking with the new flight,
    date and ticket number, then email the new e-ticket. No payment is needed
    because demo fares are constant for rebooks.
    """
    parsed = parse_travel_date(new_date, today=today)
    if parsed is None:
        return {"rebooked": False, "needs_date": True,
                "error": "I couldn't understand that date. Try 'tomorrow' or '2026-12-12'."}

    row = _schedule_row(flight_no)
    if row is None:
        return {"rebooked": False, "error": f"Unknown flight {flight_no!r}."}
    flight_no, origin, destination, depart, _arrive = row

    async with SessionLocal() as db:
        booking = (
            await db.execute(select(Booking).where(Booking.pnr == ticket_number.upper()))
        ).scalar_one_or_none()
        if booking is None:
            return {"rebooked": False, "error": f"Booking {ticket_number} was not found."}

        # Update the existing booking in place (one-way, constant price).
        booking.flight_no = flight_no
        booking.origin = origin
        booking.destination = destination
        booking.depart_date = parsed.isoformat()
        booking.depart_time = depart
        booking.status = BookingStatus.TICKETED
        booking.eticket_number = booking.eticket_number or _eticket_number(
            ticket_number)
        if email:
            booking.passenger_email = email
        if passenger_name:
            booking.passenger_name = passenger_name
        await db.commit()
        await db.refresh(booking)

    recipient = email or booking.passenger_email
    if recipient:
        result = await _issue_and_email_ticket(
            ticket_number, recipient, booking_payload(booking)
        )
        return {
            "rebooked": True,
            "ticket_number": ticket_number,
            "flight_no": flight_no,
            "route": f"{origin}-{destination}",
            "date": parsed.isoformat(),
            "depart_time": depart,
            "eticket_number": booking.eticket_number,
            "email_sent": result.get("email_sent", False),
            "email_to": recipient,
            "message": (
                f"Your flight has been rebooked to {flight_no} ({origin}-{destination}) "
                f"on {parsed.isoformat()} at {depart}. "
                f"E-ticket number: {booking.eticket_number}. "
                f"{'It has been emailed to ' + recipient if recipient else ''}"
            ),
        }

    return {
        "rebooked": True,
        "ticket_number": ticket_number,
        "flight_no": flight_no,
        "route": f"{origin}-{destination}",
        "date": parsed.isoformat(),
        "depart_time": depart,
        "eticket_number": booking.eticket_number,
        "needs_email": True,
        "message": (
            f"Your flight has been rebooked to {flight_no} ({origin}-{destination}) "
            f"on {parsed.isoformat()} at {depart}. "
            f"E-ticket number: {booking.eticket_number}. "
            "What email address should I send the new e-ticket to?"
        ),
    }


async def get_booking(pnr: str) -> Booking | None:
    async with SessionLocal() as db:
        return (
            await db.execute(select(Booking).where(Booking.pnr == pnr.upper()))
        ).scalar_one_or_none()


async def latest_successful_payment(pnr: str) -> Payment | None:
    async with SessionLocal() as db:
        return (
            await db.execute(
                select(Payment)
                .where(Payment.pnr == pnr.upper(), Payment.status == PaymentStatus.SUCCESS)
                .order_by(Payment.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()


def _eticket_number(pnr: str) -> str:
    digest = int(hashlib.sha256(
        f"eticket:{pnr}".encode()).hexdigest()[:12], 16)
    return f"074{digest % 10_000_000_000:010d}"


def _kq_logo_data_uri() -> str:
    """Base64-encoded KQ logo PNG for inline email embedding."""
    path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "assets", "kq_logo_email.svg")
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    except (OSError, IOError):
        return ""


def ticket_email(booking: Booking) -> tuple[str, str]:
    """Subject and body for the e-ticket email."""
    subject = f"Your Kenya Airways e-ticket — {booking.pnr}"
    name = booking.passenger_name or "passenger"
    body = (
        f"Dear {name},\n\n"
        "Thank you for your payment. Your Kenya Airways e-ticket is confirmed.\n\n"
        "Flight details\n"
        f"  Passenger:            {name}\n"
        f"  Booking reference:    {booking.pnr}\n"
        f"  E-ticket number:      {booking.eticket_number}\n"
        f"  Flight:               {booking.flight_no} "
        f"({booking.origin}-{booking.destination})\n"
        f"  Date:                 {booking.depart_date}\n"
        f"  Scheduled departure:  {booking.depart_time}\n"
        f"  Fare paid:            {booking.currency} {float(booking.amount or 0):,.0f}\n\n"
        "Please arrive at the airport at least two hours before departure and "
        "carry a valid photo ID.\n\n"
        "Thank you for flying Kenya Airways.\n"
    )
    return subject, body


def ticket_email_html(booking: Booking) -> str:
    """Visually appealing HTML e-ticket email body."""
    name = booking.passenger_name or "passenger"
    logo = _kq_logo_data_uri()
    logo_img = (
        f'<img src="{logo}" alt="Kenya Airways" height="40" '
        'style="display:block;margin:0 auto;border-radius:6px;" />'
        if logo
        else "<span style=\"color:#fff;font-weight:700;font-size:18px\">Kenya Airways</span>"
    )
    eticket = booking.eticket_number or "—"
    amount = f"{booking.currency} {float(booking.amount or 0):,.0f}"
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"></head>"
        '<body style="margin:0;padding:0;background:#f7f4ee;font-family:Arial,Helvetica,sans-serif;color:#141414;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f7f4ee;padding:24px 0;">'
        '<tr><td align="center">'
        '<table role="presentation" width="560" cellpadding="0" cellspacing="0" '
        'style="background:#ffffff;border-radius:14px;overflow:hidden;border:1px solid #efe9dd;">'
        # Header
        '<tr><td style="background:#e31b23;padding:24px 28px;text-align:center;">'
        + logo_img
        + '<div style="color:#ffffff;font-size:20px;font-weight:700;letter-spacing:0.5px;margin-top:10px;">Kenya Airways</div>'
        '<div style="color:rgba(255,255,255,0.85);font-size:13px;margin-top:2px;">e-Ticket Confirmation</div>'
        "</td></tr>"
        # Body
        '<tr><td style="padding:28px 32px;">'
        f'<p style="margin:0 0 12px;font-size:16px;">Dear {name},</p>'
        '<p style="margin:0 0 20px;font-size:14px;line-height:1.6;color:#3d3a36;">'
        "Thank you for your payment. Your Kenya Airways e-ticket is confirmed. "
        "Please present this email (or your e-ticket number) at check-in.</p>"
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="border:1px solid #efe9dd;border-radius:10px;overflow:hidden;font-size:14px;">'
        '<tr><td style="padding:10px 14px;background:#f7f4ee;font-weight:700;color:#3d3a36;width:40%;">Passenger</td>'
        f'<td style="padding:10px 14px;">{name}</td></tr>'
        '<tr><td style="padding:10px 14px;background:#f7f4ee;font-weight:700;color:#3d3a36;">Booking reference</td>'
        f'<td style="padding:10px 14px;font-weight:700;">{booking.pnr}</td></tr>'
        '<tr><td style="padding:10px 14px;background:#f7f4ee;font-weight:700;color:#3d3a36;">E-ticket number</td>'
        f'<td style="padding:10px 14px;">{eticket}</td></tr>'
        '<tr><td style="padding:10px 14px;background:#f7f4ee;font-weight:700;color:#3d3a36;">Flight</td>'
        f'<td style="padding:10px 14px;">{booking.flight_no} ({booking.origin}-{booking.destination})</td></tr>'
        '<tr><td style="padding:10px 14px;background:#f7f4ee;font-weight:700;color:#3d3a36;">Date</td>'
        f'<td style="padding:10px 14px;">{booking.depart_date}</td></tr>'
        '<tr><td style="padding:10px 14px;background:#f7f4ee;font-weight:700;color:#3d3a36;">Scheduled departure</td>'
        f'<td style="padding:10px 14px;">{booking.depart_time}</td></tr>'
        '<tr><td style="padding:10px 14px;background:#f7f4ee;font-weight:700;color:#3d3a36;">Fare paid</td>'
        f'<td style="padding:10px 14px;">{amount}</td></tr>'
        "</table>"
        '<p style="margin:20px 0 0;font-size:13px;line-height:1.6;color:#6f6a62;">'
        "Please arrive at the airport at least two hours before departure and "
        "carry a valid photo ID. Have a safe flight!</p>"
        "</td></tr>"
        # Footer
        '<tr><td style="padding:16px 28px;background:#f7f4ee;color:#6f6a62;font-size:12px;text-align:center;">'
        "Sent by hermes — Kenya Airways passenger support assistant.</td></tr>"
        "</table></td></tr></table></body></html>"
    )


async def issue_ticket(pnr: str, *, email: str | None = None) -> dict:
    """Issue the e-ticket for a paid booking and email it when possible.

    Agent-only: this is intentionally not exposed to the LLM tool schemas.
    """
    pnr = (pnr or "").upper()
    async with SessionLocal() as db:
        booking = (
            await db.execute(select(Booking).where(Booking.pnr == pnr))
        ).scalar_one_or_none()
        if booking is None:
            return {"ticketed": False, "error": f"Booking {pnr} was not found."}
        payment = (
            await db.execute(
                select(Payment)
                .where(Payment.pnr == pnr, Payment.status == PaymentStatus.SUCCESS)
                .order_by(Payment.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if payment is None:
            return {
                "ticketed": False,
                "error": "Payment is not verified yet; the e-ticket cannot be issued.",
            }
        if booking.status == BookingStatus.TICKETED and booking.eticket_number:
            return {
                "ticketed": True,
                "already_issued": True,
                **booking_payload(booking),
                "receipt": payment.receipt,
            }
        booking.eticket_number = booking.eticket_number or _eticket_number(pnr)
        booking.status = BookingStatus.TICKETED
        await db.commit()
        await db.refresh(booking)
        payload = booking_payload(booking)
        payload.update({"ticketed": True, "receipt": payment.receipt})

    # Email is optional in the card-first flow: the e-ticket is always delivered
    # in the conversation, and a copy is emailed only when an address is known.
    recipient = booking.passenger_email
    if not recipient:
        payload.update(
            {
                "email_sent": False,
                "email_note": "No email on file; the e-ticket was delivered in chat.",
            }
        )
        return payload

    subject, body = ticket_email(booking)
    html = ticket_email_html(booking)
    try:
        record = await asyncio.to_thread(
            email_real.send_email,
            recipient,
            subject,
            body,
            sender=settings_sender(),
            api_key=settings_key(),
            outbox_dir=settings_outbox(),
            html=html,
        )
    except Exception:
        logger.exception("E-ticket email failed for %s", pnr)
        payload.update(
            {
                "email_sent": False,
                "email_note": "The e-ticket was issued but the email could not be sent.",
            }
        )
    else:
        payload.update(
            {
                "email_sent": True,
                "email_to": recipient,
                "email_provider": record.get("provider"),
            }
        )
    return payload


def settings_sender() -> str:
    from app.config import settings

    return settings.resend_from_email


def settings_key() -> str:
    from app.config import settings

    return settings.resend_api_key


def settings_outbox() -> str:
    from app.config import settings

    return settings.email_outbox_dir


def ticket_message(ticket: dict) -> str:
    """Passenger-facing chat message for an issued e-ticket."""
    route = f"{ticket.get('origin')}-{ticket.get('destination')}"
    lines = [
        "Your Kenya Airways e-ticket is confirmed.",
        "",
        f"Booking reference: {ticket.get('pnr')}",
        f"E-ticket number: {ticket.get('eticket_number')}",
        f"Flight: {ticket.get('flight_no')} ({route})",
        f"Date: {ticket.get('date')} at {ticket.get('depart_time')}",
    ]
    if ticket.get("receipt"):
        lines.append(f"M-Pesa receipt: {ticket['receipt']}")
    if ticket.get("email_sent"):
        lines.append(f"A copy has been emailed to {ticket.get('email_to')}.")
    else:
        lines.append("Please keep this message as your e-ticket.")
    return "\n".join(lines)


def format_money(amount: float | None, currency: str = CURRENCY) -> str:
    return f"{currency} {float(amount or 0):,.0f}"


def human_date(value: str | None) -> str:
    """``2026-12-12`` -> ``12 Dec 2026`` for cards and messages."""
    if not value:
        return ""
    try:
        return date_cls.fromisoformat(value).strftime("%d %b %Y")
    except ValueError:
        return value


def _utcnow() -> datetime:
    return datetime.now(UTC)


def days_until(value: str | None) -> int | None:
    """Whole days from today to an ISO date (negative when past)."""
    if not value:
        return None
    try:
        return (date_cls.fromisoformat(value) - date_cls.today()).days
    except ValueError:
        return None


def next_weekday(day_name: str, *, today: date_cls | None = None) -> date_cls | None:
    """Next occurrence of a weekday name (inclusive of today)."""
    names = [name.lower() for name in calendar.day_name]
    try:
        target = names.index(day_name.lower())
    except ValueError:
        return None
    base = today or date_cls.today()
    ahead = (target - base.weekday()) % 7
    return base + timedelta(days=ahead)
