"""Mock rebooking service: deterministic, no network, no real reservation system.

Mirrors `services/flight_delays.py`: a plain sync function the LangGraph tool
layer calls in-process. A "rebooking" moves the passenger to the same flight
number on a new date (default: the next day) and returns a stable confirmation
reference derived from the inputs, so repeated calls for the same request are
identical. Every value here is invented prototype filler.
"""

from __future__ import annotations

import hashlib
from datetime import date as date_cls
from datetime import timedelta

from services import flight_delays


def _reference(flight_no: str, new_date: str, passenger_name: str | None) -> str:
    seed = f"hermes-rebook:{flight_no}:{new_date}:{passenger_name or 'guest'}"
    return "HRM-" + hashlib.sha256(seed.encode()).hexdigest()[:6].upper()


def rebook_flight(
    flight_no: str,
    new_date: str | None = None,
    passenger_name: str | None = None,
    reason: str | None = None,
    *,
    today: date_cls | None = None,
) -> dict:
    """Move `flight_no` to `new_date` (default: the day after today).

    Returns a JSON-able dict with `rebooked` always set; an unknown flight or
    an invalid date yields `rebooked: False` plus a `reason` the model can
    relay, instead of raising.
    """
    if not flight_no:
        return {"rebooked": False, "reason": "No flight number was provided."}

    base = today or date_cls.today()
    original = flight_delays.get_flight_status(flight_no, base.isoformat())
    if not original.get("found"):
        return {
            "rebooked": False,
            "reason": f"No flight {original.get('flight_no', flight_no)} in the KQ schedule.",
        }

    if new_date:
        try:
            target = date_cls.fromisoformat(flight_delays.canonical_date(new_date))
        except ValueError as exc:
            return {"rebooked": False, "reason": str(exc)}
    else:
        target = base + timedelta(days=1)
    target_iso = target.isoformat()

    moved = flight_delays.get_flight_status(original["flight_no"], target_iso)
    reference = _reference(original["flight_no"], target_iso, passenger_name)
    return {
        "rebooked": True,
        "reference": reference,
        "original_flight_no": original["flight_no"],
        "flight_no": moved["flight_no"],
        "route": moved["route"],
        "origin": moved["origin"],
        "destination": moved["destination"],
        "original_date": original["date"],
        "date": target_iso,
        "scheduled_departure": moved["scheduled_departure"],
        "scheduled_arrival": moved["scheduled_arrival"],
        "status": moved["status"],
        "passenger_name": passenger_name,
        "reason": reason,
        "message": (
            f"Rebooked {original['flight_no']} to {target_iso}. "
            f"Confirmation reference {reference}."
        ),
    }
