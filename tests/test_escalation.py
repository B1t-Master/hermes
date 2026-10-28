"""Soft-escalation routing: hard triggers escalate at once, soft ones ask first."""

import unittest
from unittest import mock

from app.agents import graph


async def _no_triggers(_state):
    return ""


class RouteTest(unittest.IsolatedAsyncioTestCase):
    async def _route(self, **state):
        with mock.patch.object(graph, "_history_triggers", _no_triggers):
            return await graph.route(state)

    async def test_safety_escalates_immediately(self):
        result = await self._route(intent="safety", intent_source="keyword")
        self.assertTrue(result["escalate"])
        self.assertFalse(result.get("offer_escalation"))
        self.assertEqual(result["escalation_reason"], "safety keyword trigger")

    async def test_human_request_escalates_immediately(self):
        result = await self._route(intent="human_request")
        self.assertTrue(result["escalate"])
        self.assertFalse(result.get("offer_escalation"))

    async def test_compensation_escalates_by_default(self):
        result = await self._route(intent="compensation")
        self.assertTrue(result["escalate"])
        self.assertFalse(result.get("offer_escalation"))

    async def test_compensation_offers_when_configured_soft(self):
        with mock.patch.object(graph.settings, "escalate_on_compensation", False):
            result = await self._route(intent="compensation", soft_offers=0)
        self.assertFalse(result.get("escalate"))
        self.assertTrue(result["offer_escalation"])

    async def test_soft_trigger_offers_before_escalating(self):
        async def triggers(_state):
            return "unresolved for more than 3 turns"

        with mock.patch.object(graph, "_history_triggers", triggers):
            result = await graph.route({"intent": "routine", "soft_offers": 0})
        self.assertFalse(result.get("escalate"))
        self.assertTrue(result["offer_escalation"])
        self.assertEqual(result["escalation_reason"], "unresolved for more than 3 turns")

    async def test_soft_trigger_escalates_after_max_offers(self):
        async def triggers(_state):
            return "unresolved for more than 3 turns"

        with (
            mock.patch.object(graph, "_history_triggers", triggers),
            mock.patch.object(graph.settings, "soft_escalation_max_offers", 2),
        ):
            result = await graph.route({"intent": "routine", "soft_offers": 2})
        self.assertTrue(result["escalate"])
        self.assertFalse(result.get("offer_escalation"))

    async def test_soft_disabled_escalates_immediately(self):
        async def triggers(_state):
            return "sentiment below -0.75 for 2 consecutive turns"

        with (
            mock.patch.object(graph, "_history_triggers", triggers),
            mock.patch.object(graph.settings, "soft_escalation_enabled", False),
        ):
            result = await graph.route({"intent": "routine", "soft_offers": 0})
        self.assertTrue(result["escalate"])
        self.assertFalse(result.get("offer_escalation"))

    async def test_routine_stays_routine(self):
        result = await self._route(intent="routine", soft_offers=0)
        self.assertFalse(result.get("escalate"))
        self.assertFalse(result.get("offer_escalation"))
        self.assertEqual(result["escalation_reason"], "")


class RouteNextTest(unittest.TestCase):
    def test_precedence(self):
        self.assertEqual(graph._route_next({"escalate": True}), "escalate")
        self.assertEqual(graph._route_next({"offer_escalation": True}), "offer")
        self.assertEqual(graph._route_next({}), "retrieve")
        # Escalate wins over a stale offer flag.
        self.assertEqual(
            graph._route_next({"escalate": True, "offer_escalation": True}), "escalate"
        )


class OfferOptionsTest(unittest.TestCase):
    def test_options_are_labelled_pairs(self):
        for option in graph.SOFT_ESCALATION_OPTIONS:
            self.assertIn("label", option)
            self.assertIn("value", option)
            self.assertTrue(option["label"])
            self.assertTrue(option["value"])


if __name__ == "__main__":
    unittest.main()
