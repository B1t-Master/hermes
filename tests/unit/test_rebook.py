import random
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from services import rebook

TICKET = "1234567890"


def _rebook(**kwargs):
    kwargs.setdefault("ticket_number", TICKET)
    kwargs.setdefault("today", date(2025, 3, 1))
    kwargs.setdefault("rng", random.Random(1))
    return rebook.rebook_flight(**kwargs)


class RebookFlightTest(unittest.TestCase):
    def test_picks_a_flight_from_the_schedule(self):
        result = _rebook(new_date="tomorrow")
        self.assertTrue(result["rebooked"])
        self.assertEqual(result["ticket_number"], TICKET)
        self.assertEqual(result["date"], "2025-03-02")
        self.assertTrue(result["reference"].startswith("HRM-"))
        self.assertRegex(result["flight_no"], r"^KQ\d+$")

    def test_new_date_is_required(self):
        result = rebook.rebook_flight(TICKET, today=date(2025, 3, 1))
        # Positional first arg is the flight number, so no ticket was supplied.
        self.assertTrue(result["needs_ticket"])

        result = rebook.rebook_flight(
            ticket_number=TICKET, today=date(2025, 3, 1)
        )
        self.assertFalse(result["rebooked"])
        self.assertTrue(result["needs_date"])

    def test_explicit_iso_date(self):
        result = _rebook(new_date="2025-04-10")
        self.assertTrue(result["rebooked"])
        self.assertEqual(result["date"], "2025-04-10")

    def test_natural_language_dates(self):
        cases = {
            "tomorrow": "2025-03-02",
            "tomorrow afternoon": "2025-03-02",
            "Tomorrow morning": "2025-03-02",
            "day after tomorrow": "2025-03-03",
            "today": "2025-03-01",
        }
        for phrase, expected in cases.items():
            result = _rebook(new_date=phrase)
            self.assertTrue(result["rebooked"], phrase)
            self.assertEqual(result["date"], expected, phrase)

    def test_reference_is_deterministic_for_same_inputs(self):
        first = _rebook(new_date="2025-03-02")
        second = _rebook(new_date="2025-03-02")
        self.assertEqual(first["reference"], second["reference"])

    def test_reference_varies_with_ticket(self):
        first = _rebook(new_date="2025-03-02")
        second = _rebook(ticket_number="9999999999", new_date="2025-03-02")
        self.assertNotEqual(first["reference"], second["reference"])

    def test_invalid_date_is_not_rebooked(self):
        result = _rebook(new_date="not-a-date")
        self.assertFalse(result["rebooked"])
        self.assertTrue(result["date_invalid"])

    def test_missing_ticket_asks_for_it(self):
        result = rebook.rebook_flight(
            "KQ100", new_date="tomorrow", today=date(2025, 3, 1)
        )
        self.assertFalse(result["rebooked"])
        self.assertTrue(result["needs_ticket"])
        self.assertIn("booking reference", result["reason"].lower())

    def test_rebooking_is_recorded_in_the_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "flights.sqlite")
            result = _rebook(new_date="2025-03-02", db_path=db)
            conn = sqlite3.connect(db)
            try:
                row = conn.execute(
                    "SELECT flight_no, new_date, reference FROM rebookings"
                ).fetchone()
            finally:
                conn.close()
        self.assertEqual(row[0], result["flight_no"])
        self.assertEqual(row[1], "2025-03-02")
        self.assertEqual(row[2], result["reference"])


class ConfirmationEmailTest(unittest.TestCase):
    def test_subject_and_body_carry_the_details(self):
        result = _rebook(new_date="2025-03-02")
        subject, body = rebook.confirmation_email(result)
        self.assertIn(result["reference"], subject)
        for expected in (result["flight_no"], TICKET, result["reference"], "2025-03-02"):
            self.assertIn(expected, body)


if __name__ == "__main__":
    unittest.main()
