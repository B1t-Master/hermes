import unittest
from datetime import date

from services import rebook

TICKET = "1234567890"


class RebookFlightTest(unittest.TestCase):
    def test_default_moves_to_next_day(self):
        result = rebook.rebook_flight("KQ100", TICKET, today=date(2025, 3, 1))
        self.assertTrue(result["rebooked"])
        self.assertEqual(result["original_flight_no"], "KQ100")
        self.assertEqual(result["flight_no"], "KQ100")
        self.assertEqual(result["original_date"], "2025-03-01")
        self.assertEqual(result["date"], "2025-03-02")
        self.assertEqual(result["route"], "NBO-LHR")
        self.assertEqual(result["ticket_number"], TICKET)
        self.assertTrue(result["reference"].startswith("HRM-"))

    def test_reference_is_deterministic(self):
        first = rebook.rebook_flight("KQ100", TICKET, today=date(2025, 3, 1))
        second = rebook.rebook_flight("KQ100", TICKET, today=date(2025, 3, 1))
        self.assertEqual(first["reference"], second["reference"])

    def test_reference_varies_with_ticket(self):
        first = rebook.rebook_flight("KQ100", TICKET, today=date(2025, 3, 1))
        second = rebook.rebook_flight("KQ100", "9999999999", today=date(2025, 3, 1))
        self.assertNotEqual(first["reference"], second["reference"])

    def test_explicit_new_date(self):
        result = rebook.rebook_flight("KQ720", TICKET, new_date="2025-04-10")
        self.assertTrue(result["rebooked"])
        self.assertEqual(result["date"], "2025-04-10")
        self.assertEqual(result["route"], "NBO-CMB")

    def test_unknown_flight_is_not_rebooked(self):
        result = rebook.rebook_flight("KQ999", TICKET)
        self.assertFalse(result["rebooked"])
        self.assertIn("KQ999", result["reason"])

    def test_invalid_date_is_not_rebooked(self):
        result = rebook.rebook_flight("KQ100", TICKET, new_date="not-a-date")
        self.assertFalse(result["rebooked"])

    def test_missing_flight_no_is_not_rebooked(self):
        result = rebook.rebook_flight("", TICKET)
        self.assertFalse(result["rebooked"])

    def test_missing_ticket_asks_for_it(self):
        result = rebook.rebook_flight("KQ100", today=date(2025, 3, 1))
        self.assertFalse(result["rebooked"])
        self.assertTrue(result["needs_ticket"])
        self.assertIn("booking reference", result["reason"].lower())


class ConfirmationEmailTest(unittest.TestCase):
    def test_subject_and_body_carry_the_details(self):
        result = rebook.rebook_flight("KQ100", TICKET, today=date(2025, 3, 1))
        subject, body = rebook.confirmation_email(result)
        self.assertIn(result["reference"], subject)
        for expected in ("KQ100", TICKET, result["reference"], "2025-03-02"):
            self.assertIn(expected, body)


if __name__ == "__main__":
    unittest.main()
