"""Baggage guidance: deterministic detection plus the three arrival options."""

import unittest
from unittest import mock

from app.agents import baggage, graph


class DetectBaggageTest(unittest.TestCase):
    def test_matches_lost_and_delayed_bag_queries(self):
        for query in (
            "my bag is missing",
            "I lost my luggage",
            "my suitcase didn't arrive",
            "delayed baggage claim",
            "where do I collect my bags?",
            "how can I track my baggage?",
            "is there a worldtracer reference for my bag?",
            "where is the lost and found desk?",
            "my bag never showed up",
        ):
            self.assertTrue(baggage.detect_baggage_request(query), query)

    def test_ignores_general_baggage_questions(self):
        for query in (
            "what is my checked baggage allowance?",
            "can I take a suitcase on board?",
            "is the self rebooking service free?",
            "how do I check in?",
            "hello",
        ):
            self.assertFalse(baggage.detect_baggage_request(query), query)


class GuidanceContentTest(unittest.TestCase):
    def test_renders_all_three_contact_points(self):
        with (
            mock.patch.object(baggage.settings, "baggage_arrival_claim", "CLAIM-HALL"),
            mock.patch.object(baggage.settings, "baggage_lost_and_found", "LNF-DESK"),
            mock.patch.object(
                baggage.settings, "worldtracer_url", "https://wt.example/"
            ),
        ):
            text = baggage.render_guidance()
        self.assertIn("CLAIM-HALL", text)
        self.assertIn("LNF-DESK", text)
        self.assertIn("https://wt.example/", text)
        self.assertEqual(len(baggage.guidance_lines()), 3)


class BaggageNodeTest(unittest.IsolatedAsyncioTestCase):
    async def test_sets_context_for_baggage_query(self):
        with mock.patch.object(graph, "detect_baggage_request", return_value=True):
            result = await graph.baggage_lookup({"query": "my bag is missing"})
        self.assertIn("baggage_context", result)
        self.assertTrue(result["baggage_context"])

    async def test_skips_unrelated_query(self):
        with mock.patch.object(graph, "detect_baggage_request", return_value=False):
            result = await graph.baggage_lookup({"query": "what is my allowance?"})
        self.assertEqual(result, {})

    async def test_answer_options_are_labelled_pairs(self):
        for option in graph.BAGGAGE_OPTIONS:
            self.assertIn("label", option)
            self.assertIn("value", option)
            self.assertTrue(option["label"])
            self.assertTrue(option["value"])


if __name__ == "__main__":
    unittest.main()
