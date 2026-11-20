"""M-Pesa payment orchestration for the in-chat booking flow.

The passenger enters their number in a payment card, the browser posts it to
``POST /chat/payments/{id}/phone``, and this package sends the STK push. A short
background poller watches Daraja (and the simulated provider) until the payment
resolves. On success the booking is marked paid, the passenger gets a bot
message, and the dashboard is told to update.

The number is only ever handled here and in the payment endpoints: it is never
part of an LLM prompt or a chat message.
"""

from .polling import resume_pending_payments, schedule_poller, shutdown_payments
from .results import apply_payment_result, apply_payment_result_by_checkout
from .start import start_payment
from .state import (
    PaymentError,
    broadcast_payment_update,
    get_payment_payload,
    load_payment,
    payment_payload,
)

__all__ = [
    "PaymentError",
    "apply_payment_result",
    "apply_payment_result_by_checkout",
    "broadcast_payment_update",
    "get_payment_payload",
    "load_payment",
    "payment_payload",
    "resume_pending_payments",
    "schedule_poller",
    "shutdown_payments",
    "start_payment",
]
