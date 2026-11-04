"""Standalone MCP server for bookings, rebookings, and tracking operations.

Run it directly (stdio transport):
    .venv/bin/python -m services.ops_mcp

Or configure in MCP clients as "hermes-ops".

Tools exposed:
  - rebook_flight(ticket_number, new_date, ...)   wraps services.rebook
  - search_flights(origin, destination, date)     wraps services.booking
  - create_booking(flight_no, date, ...)          held booking + payment row
  - issue_ticket(pnr)                             paid booking -> e-ticket
  - mpesa_stk_push(phone, amount, pnr)            Daraja / simulated STK push
  - verify_payment(pnr, checkout_request_id)      one-shot status lookup

`issue_ticket`, `mpesa_stk_push` and `verify_payment` are deliberately not in
the LLM tool schema: the passenger pays through the in-chat payment card and a
human agent issues the ticket. They exist here so every booking operation has
one MCP entry point.
"""

from mcp.server.fastmcp import FastMCP
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Payment, PaymentStatus
from services import booking, mpesa, rebook

mcp = FastMCP(
    "hermes-ops",
    instructions=(
        "Kenya Airways booking and rebooking operations (prototype). "
        "Rebookings and bookings wrap the mock schedule; payments use Daraja "
        "when configured and a deterministic simulator otherwise."
    ),
)


@mcp.tool()
def rebook_flight(
    ticket_number: str | None = None,
    new_date: str | None = None,
    flight_no: str | None = None,
    passenger_name: str | None = None,
    reason: str | None = None,
) -> dict:
    """Rebook a passenger onto an available flight for a new travel date.

    Args:
        ticket_number: Booking reference or e-ticket number (required)
        new_date: New travel date (tomorrow, YYYY-MM-DD, etc., required)
        flight_no: Optional original flight number
        passenger_name: Optional passenger name
        reason: Optional reason for rebooking
    """
    return rebook.rebook_flight(
        flight_no=flight_no,
        ticket_number=ticket_number,
        new_date=new_date,
        passenger_name=passenger_name,
        reason=reason,
    )


@mcp.tool()
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search the mock schedule for flights on one route and date.

    Args:
        origin: City name or IATA code, e.g. "Nairobi" or "NBO"
        destination: City name or IATA code, e.g. "Mombasa" or "MBA"
        date: Travel date, e.g. "2026-12-12", "12 December" or "tomorrow"
    """
    return booking.search_flights(origin, destination, date)


@mcp.tool()
async def create_booking(
    flight_no: str,
    date: str,
    passenger_name: str | None = None,
    passenger_email: str | None = None,
    passenger_id: str | None = None,
    conversation_id: str | None = None,
) -> dict:
    """Create a held booking and a pending M-Pesa payment row.

    The in-chat payment card collects the M-Pesa number and starts the STK
    push, so no phone number is accepted or prompted here.

    Args:
        flight_no: Flight number returned by search_flights, e.g. "KQ444".
        date: Travel date, e.g. "2026-12-12".
        passenger_name: Passenger name when known.
        passenger_email: Contact email when known (optional; can be added after payment).
        passenger_id: Internal passenger id (injected by the backend).
        conversation_id: Internal conversation id (injected by the backend).
    """
    return await booking.create_booking(
        flight_no,
        date,
        passenger_name=passenger_name,
        passenger_email=passenger_email,
        passenger_id=passenger_id,
        conversation_id=conversation_id,
    )


@mcp.tool()
async def issue_ticket(pnr: str, email: str | None = None) -> dict:
    """Issue the e-ticket for a paid booking (agent-only operation)."""
    return await booking.issue_ticket(pnr, email=email)


@mcp.tool()
async def mpesa_stk_push(phone: str, amount: float, pnr: str) -> dict:
    """Send an M-Pesa STK push for a booking payment.

    This is a low-level entry point. The web flow uses the payment card and
    the background poller in ``app.agents.payments`` instead.
    """
    try:
        normalised = mpesa.normalize_phone(phone)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    async with SessionLocal() as db:
        payment = (
            await db.execute(
                select(Payment)
                .where(Payment.pnr == pnr.upper(), Payment.status == PaymentStatus.PENDING)
                .order_by(Payment.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if payment is None:
            return {"ok": False, "error": f"No pending payment for {pnr}."}
        try:
            response = await mpesa.get_client().stk_push(
                normalised, amount, payment.pnr, f"Kenya Airways {payment.pnr}"
            )
        except mpesa.MpesaError as exc:
            return {"ok": False, "error": str(exc)}
        payment.phone = normalised
        payment.checkout_request_id = response.get("checkout_request_id")
        payment.merchant_request_id = response.get("merchant_request_id")
        payment.raw = response.get("raw")
        await db.commit()
        return {"ok": True, "payment_id": str(payment.id), **response}


@mcp.tool()
async def verify_payment(
    pnr: str | None = None, checkout_request_id: str | None = None
) -> dict:
    """Return the current M-Pesa payment status for a booking or checkout id."""
    async with SessionLocal() as db:
        stmt = select(Payment).order_by(Payment.created_at.desc()).limit(1)
        if checkout_request_id:
            stmt = stmt.where(Payment.checkout_request_id == checkout_request_id)
        elif pnr:
            stmt = stmt.where(Payment.pnr == pnr.upper())
        else:
            return {"ok": False, "error": "Pass a pnr or checkout_request_id."}
        payment = (await db.execute(stmt)).scalar_one_or_none()
        if payment is None:
            return {"ok": False, "error": "Payment not found."}
        if payment.checkout_request_id and payment.status == PaymentStatus.PENDING:
            try:
                probe = await mpesa.get_client().query(payment.checkout_request_id)
            except mpesa.MpesaError as exc:
                probe = {"error": str(exc)}
        else:
            probe = {}
    return {
        "ok": True,
        "payment_id": str(payment.id),
        "pnr": payment.pnr,
        "status": payment.status.value,
        "receipt": payment.receipt,
        "checkout_request_id": payment.checkout_request_id,
        "probe": probe,
    }


@mcp.tool()
async def confirm_rebook(
    ticket_number: str,
    new_date: str,
    flight_no: str,
    email: str | None = None,
    passenger_name: str | None = None,
) -> dict:
    """Confirm a rebooking: update the existing booking and email the new ticket.

    Args:
        ticket_number: Booking reference or e-ticket number.
        new_date: New travel date.
        flight_no: Flight number chosen from the alternatives.
        email: Contact email for the new e-ticket.
        passenger_name: Passenger name when known.
    """
    return await booking.confirm_rebook(
        ticket_number,
        new_date,
        flight_no,
        email=email,
        passenger_name=passenger_name,
    )


if __name__ == "__main__":
    mcp.run()
