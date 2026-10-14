"""Real email delivery via Resend, with the mock outbox as a safety net.

`send_email` prefers Resend when an API key is configured, and otherwise (or
if the API call fails) writes to the mock outbox so a delivery problem never
breaks the passenger's turn. The returned record always carries a `provider`
field — ``resend``, ``mock`` (no key), or ``mock-fallback`` (API error) — so
callers and tests can tell what actually happened.

The Resend SDK is synchronous, so callers that run on the event loop should
invoke this through a thread (see ``app.agents.tools``).
"""

from __future__ import annotations

import html
import logging
import os
from datetime import UTC, datetime

from services import email as mock_email

logger = logging.getLogger(__name__)

DEFAULT_FROM = "onboarding@resend.dev"


def _to_html(body: str) -> str:
    """Minimal HTML rendering: escaped text, line breaks preserved."""
    escaped = html.escape(body)
    return (
        "<div style=\"white-space:pre-wrap;font-family:sans-serif;"
        'font-size:14px;line-height:1.5">' + escaped + "</div>"
    )


def send_email(
    to: str,
    subject: str,
    body: str,
    *,
    sender: str | None = None,
    api_key: str | None = None,
    outbox_dir: str | None = None,
) -> dict:
    """Send ``body`` to ``to`` via Resend; fall back to the mock outbox.

    ``sender`` / ``api_key`` / ``outbox_dir`` default to the corresponding
    environment variables when not passed explicitly.
    """
    sender = sender or os.environ.get("RESEND_FROM_EMAIL") or DEFAULT_FROM
    api_key = api_key if api_key is not None else os.environ.get("RESEND_API_KEY", "")

    if not api_key:
        record = mock_email.send_email(
            to, subject, body, sender=sender, outbox_dir=outbox_dir
        )
        record["provider"] = "mock"
        return record

    try:
        import resend

        resend.api_key = api_key
        result = resend.Emails.send(
            {
                "from": sender,
                "to": [to],
                "subject": subject,
                "text": body,
                "html": _to_html(body),
            }
        )
        logger.info("Resend accepted email id=%s to=%s", (result or {}).get("id"), to)
        return {
            "to": to,
            "from": sender,
            "subject": subject,
            "provider": "resend",
            "id": (result or {}).get("id"),
            "sent_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    except Exception:
        logger.exception("Resend delivery failed; writing to mock outbox instead")
        record = mock_email.send_email(
            to, subject, body, sender=sender, outbox_dir=outbox_dir
        )
        record["provider"] = "mock-fallback"
        return record
