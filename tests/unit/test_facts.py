"""Conversation memory: fact extraction and the known-context prompt block."""

import unittest

from app.agents import graph
from app.agents.facts import (
    apply_rebook_result,
    detect_human_decline,
    extract_facts,
)


class ExtractFactsTest(unittest.TestCase):
    def test_extracts_long_ticket_number(self):
        facts = extract_facts("My ticket is 2971234567890")
        self.assertEqual(facts["ticket_number"], "2971234567890")

    def test_extracts_hyphenated_reference(self):
        facts = extract_facts("the ref is HRM-9B4770")
        self.assertEqual(facts["ticket_number"], "HRM-9B4770")

    def test_extracts_hinted_reference(self):
        facts = extract_facts("my booking reference is ABC123")
        self.assertEqual(facts["ticket_number"], "ABC123")

    def test_extracts_email_lowercased(self):
        facts = extract_facts("email me at Test.User@Example.COM please")
        self.assertEqual(facts["email"], "test.user@example.com")

    def test_extracts_kenyan_phone(self):
        facts = extract_facts("call me on 0712 345 678")
        self.assertEqual(facts["phone"], "0712345678")

    def test_extracts_relative_date(self):
        facts = extract_facts("rebook me for tomorrow afternoon")
        self.assertEqual(facts["new_date"], "tomorrow")

    def test_iso_date(self):
        facts = extract_facts("move it to 2026-04-20")
        self.assertEqual(facts["new_date"], "2026-04-20")

    def test_tracking_reference_is_not_a_booking_reference(self):
        facts = extract_facts("my worldtracer reference is 1234567890")
        self.assertEqual(facts.get("tracking_ref"), "1234567890")
        self.assertNotIn("ticket_number", facts)

    def test_merges_without_overwriting_known_slots(self):
        first = extract_facts("my ticket is 1234567890")
        second = extract_facts("tomorrow please", first)
        self.assertEqual(second["ticket_number"], "1234567890")
        self.assertEqual(second["new_date"], "tomorrow")

    def test_plain_message_leaves_slots_empty(self):
        self.assertEqual(extract_facts("hello there"), {})

    def test_empty_message_returns_existing(self):
        existing = {"ticket_number": "1234567890"}
        self.assertEqual(extract_facts("", existing), existing)

    def test_confirmation_reference_does_not_overwrite_the_ticket(self):
        """The rebooking confirmation is not the passenger's ticket number.

        The bot answers a rebooking with "Your new confirmation reference is
        HRM-D344AC". Scanning that message used to overwrite the passenger's
        e-ticket, so "what is my ticket number?" came back with the wrong
        code.
        """
        facts = extract_facts("My ticket number is HRM-1234567")
        facts = extract_facts(
            "Your flight has been re-booked to KQ312 (NBO - CPT) on 2026-10-22. "
            "Your new confirmation reference is HRM-D344AC.",
            facts,
            source="assistant",
        )
        self.assertEqual(facts["ticket_number"], "HRM-1234567")
        self.assertEqual(facts["confirmation_ref"], "HRM-D344AC")

    def test_assistant_message_never_overwrites_a_known_ticket(self):
        facts = extract_facts("My ticket number is HRM-1234567")
        facts = extract_facts(
            "Your booking HRM-1234567 is now on KQ312.", facts, source="assistant"
        )
        self.assertEqual(facts["ticket_number"], "HRM-1234567")

    def test_assistant_confirmation_alone_never_becomes_a_ticket(self):
        facts = extract_facts(
            "Confirmation reference HRM-D344AC.", {}, source="assistant"
        )
        self.assertNotIn("ticket_number", facts)
        self.assertEqual(facts["confirmation_ref"], "HRM-D344AC")

    def test_non_breaking_hyphen_reference_is_captured(self):
        """Models emit U+2011 in references: HRM-D344AC."""
        facts = extract_facts("my ticket is HRM-1234567")
        self.assertEqual(facts["ticket_number"], "HRM-1234567")

    def test_compact_reference_is_captured(self):
        facts = extract_facts("my ticket is HRM9B4770")
        self.assertEqual(facts["ticket_number"], "HRM9B4770")

    def test_phone_number_is_not_a_ticket(self):
        facts = extract_facts("call me on 0712345678")
        self.assertNotIn("ticket_number", facts)
        self.assertEqual(facts["phone"], "0712345678")

    def test_detects_a_declined_handoff(self):
        self.assertTrue(detect_human_decline("No thanks, I'll continue with the chat."))
        self.assertTrue(detect_human_decline("no thank you"))
        self.assertFalse(detect_human_decline("I need to change my flight"))


class ApplyRebookResultTest(unittest.TestCase):
    def test_tool_result_is_authoritative(self):
        facts = apply_rebook_result(
            {"ticket_number": "HRM-1234567"},
            {
                "rebooked": True,
                "ticket_number": "HRM-1234567",
                "reference": "HRM-D344AC",
                "flight_no": "KQ312",
                "date": "2026-10-22",
                "route": "NBO-CPT",
            },
        )
        self.assertEqual(facts["ticket_number"], "HRM-1234567")
        self.assertEqual(facts["confirmation_ref"], "HRM-D344AC")
        self.assertEqual(facts["new_flight_no"], "KQ312")
        self.assertEqual(facts["new_date"], "2026-10-22")
        self.assertEqual(facts["route"], "NBO-CPT")

    def test_a_failed_rebooking_changes_nothing(self):
        facts = {"ticket_number": "HRM-1234567"}
        self.assertEqual(apply_rebook_result(facts, {"rebooked": False}), facts)


class KnownContextBlockTest(unittest.TestCase):
    def test_empty_state_renders_nothing(self):
        self.assertEqual(graph._known_context_block({}), "")

    def test_includes_facts_and_profile(self):
        block = graph._known_context_block(
            {
                "facts": {"ticket_number": "1234567890", "phone": "0712345678"},
                "profile": {"name": "Amina Yusuf", "email": "amina@example.com"},
            }
        )
        self.assertIn("Amina Yusuf", block)
        self.assertIn("amina@example.com", block)
        self.assertIn("1234567890", block)
        self.assertIn("0712345678", block)
        self.assertIn("never ask for them again", block)

    def test_declined_handoff_is_visible_to_the_model(self):
        block = graph._known_context_block({"facts": {"human_declined": True}})
        self.assertIn("declined a human hand-off", block)

    def test_anonymous_name_is_omitted(self):
        block = graph._known_context_block(
            {"profile": {"name": "Guest", "anonymous": True}}
        )
        self.assertEqual(block, "")

    def test_lists_inherited_bookings(self):
        block = graph._known_context_block(
            {
                "profile": {
                    "bookings": [
                        {
                            "pnr": "KQ7X2B9",
                            "flight_no": "KQ100",
                            "origin": "NBO",
                            "destination": "LHR",
                            "date": "2026-05-01",
                            "status": "ticketed",
                        }
                    ]
                }
            }
        )
        self.assertIn("KQ7X2B9", block)
        self.assertIn("NBO-LHR", block)


if __name__ == "__main__":
    unittest.main()
