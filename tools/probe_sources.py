"""Probe candidate KQ pages: how much real content does each one give?

Run:  .venv/bin/python tools/probe_sources.py
Writes tools/probe_report.json with the verdict for every candidate URL.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

from ingest.parser import boilerplate_ratio, scrape_url  # noqa: E402

BASE = "https://www.kenya-airways.com/en-ke/"

CANDIDATES = [
    "plan/baggage-information/baggage-allowance/",
    "plan/baggage-information/hand-baggage/",
    "plan/baggage-information/delayed-baggage/",
    "plan/baggage-information/lost-and-found-items/",
    "plan/baggage-information/baggage-restrictions/",
    "plan/baggage-information/special-baggage/",
    "plan/baggage-information/conditions-of-carriage-baggage/",
    "plan/baggage-information/pet-carriage/",
    "plan/baggage-information/dangerous-goods/",
    "book-manage/post-booking/check-in/",
    "book-manage/post-booking/check-in/baggage-drop-off/",
    "book-manage/post-booking/travel-documents/",
    "experience/at-the-airport/baggage-drop-off/",
    "experience/at-the-airport/kiosk-check-in/",
    "experience/at-the-airport/lounge-services/",
    "experience/at-the-airport/boarding/",
    "book-manage/manage-booking/change-booking/",
    "book-manage/manage-booking/cancel-flight/",
    "book-manage/manage-booking/online-refunds/",
    "book-manage/manage-booking/request-for-your-refund/",
    "book-manage/manage-booking/free-change-form/",
    "book-manage/manage-booking/upgrade-flight/",
    "book-manage/manage-booking/",
    "book-manage/self-rebooking-service/",
    "book-manage/search-book/flight-timetable/",
    "policies/24-hours-refund-policy/",
    "policies/",
    "help/your-trip/flight-delay-and-cancellations/",
    "help/your-trip/baggage-track-and-trace/",
    "help/your-trip/cancel-booking/",
    "help/your-trip/change-booking/",
    "help/your-trip/seats/",
    "help/faqs/baggage-restrictions/",
    "help/faqs/compensation-on-delayed-damaged-or-lost-baggage/",
    "help/faqs/delayed-baggage/",
    "help/faqs/travelling-with-an-infant/",
    "help/faqs/extra-legroom-seat/",
    "help/faqs/",
    "help/customer-help/",
    "plan/special-care/infant-toddlers/",
    "plan/special-care/unaccompanied-minors/",
    "plan/travel-information/travel-requirements/",
    "plan/travel-information/",
    "contact/",
]


def main() -> int:
    report = []
    for path in CANDIDATES:
        url = BASE + path
        entry = {"url": url, "path": path}
        try:
            text = scrape_url(url, timeout=25)
        except Exception as exc:  # noqa: BLE001
            entry.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            report.append(entry)
            print(f"ERR  {path}  {entry['error']}", flush=True)
            continue
        length = len(text.strip())
        junk = boilerplate_ratio(text)
        entry.update(
            {
                "ok": True,
                "chars": length,
                "boilerplate": round(junk, 3),
                "has_table": "Table data:" in text,
                "preview": re.sub(r"\s+", " ", text)[:220],
            }
        )
        report.append(entry)
        flag = "OK " if length >= 400 and junk <= 0.5 else "THIN"
        print(f"{flag} {length:>6} chars  junk={junk:.0%}  {path}", flush=True)

    out = Path("tools/probe_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
