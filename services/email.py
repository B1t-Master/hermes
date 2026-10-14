"""Mock email sender: logs each message and writes it to an outbox directory.

Real delivery (SMTP or a provider API) is out of scope for the prototype, so
"sent" mail is persisted under `EMAIL_OUTBOX_DIR` (default `.mock_emails/`) and
logged. Replacing this one function with a real provider is all a production
swap needs.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OUTBOX_DIR = os.environ.get("EMAIL_OUTBOX_DIR", ".mock_emails")
DEFAULT_SENDER = "hermes@kenya-airways.example"

_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")[:60] or "message"


def send_email(
    to: str,
    subject: str,
    body: str,
    *,
    sender: str = DEFAULT_SENDER,
    outbox_dir: str | None = None,
) -> dict:
    """Pretend to send an email and return the delivery record.

    The outbox write is best effort: a filesystem error is logged and the turn
    continues, since a mock email must never break the passenger's answer.
    """
    sent_at = datetime.now(UTC).isoformat(timespec="seconds")
    record: dict = {
        "to": to,
        "from": sender,
        "subject": subject,
        "body": body,
        "sent_at": sent_at,
    }
    logger.info("MOCK EMAIL to=%s subject=%r\n%s", to, subject, body)

    directory = Path(outbox_dir or DEFAULT_OUTBOX_DIR)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        path = directory / f"{stamp}-{_slug(subject)}.txt"
        path.write_text(
            f"To: {to}\nFrom: {sender}\nSubject: {subject}\nSent: {sent_at}\n\n{body}\n",
            encoding="utf-8",
        )
        record["path"] = str(path)
    except OSError:
        logger.exception("Could not write mock email to %s", directory)

    return record
