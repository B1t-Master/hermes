"""Escalation routing: hard triggers escalate at once, soft ones answer first.

A soft trigger (compensation, a negative-sentiment streak, a conversation the
bot cannot close) no longer replaces the answer: the bot answers the passenger
and attaches a "Connect me to a human" quick reply. Only after
`soft_escalation_max_offers` unanswered offers does it hand off.
"""

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

    async def test_compensation_offers_before_escalating(self):
        """A refund question gets an answer first, not an instant hand-off."""
        result = await self._route(intent="compensation", soft_offers=0)
        self.assertFalse(result.get("escalate"))
        self.assertTrue(result["offer_escalation"])

    async def test_compensation_escalates_when_configured_hard(self):
        with mock.patch.object(graph.settings, "escalate_on_compensation", True):
            result = await self._route(intent="compensation")
        self.assertTrue(result["escalate"])
        self.assertFalse(result.get("offer_escalation"))

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

    async def test_declined_handoff_is_not_offered_again(self):
        """Once the passenger says "no thanks", the bot stops pushing."""

        async def triggers(_state):
            return "unresolved for more than 3 turns"

        with mock.patch.object(graph, "_history_triggers", triggers):
            result = await graph.route(
                {"intent": "routine", "soft_offers": 0, "facts": {"human_declined": True}}
            )
        self.assertFalse(result.get("escalate"))
        self.assertFalse(result.get("offer_escalation"))

    async def test_routine_stays_routine(self):
        result = await self._route(intent="routine", soft_offers=0)
        self.assertFalse(result.get("escalate"))
        self.assertFalse(result.get("offer_escalation"))
        self.assertEqual(result["escalation_reason"], "")


class RouteNextTest(unittest.TestCase):
    def test_precedence(self):
        self.assertEqual(graph._route_next({"escalate": True}), "escalate")
        self.assertEqual(graph._route_next({}), "retrieve")
        # A soft offer does not divert the turn: the answer comes first.
        self.assertEqual(graph._route_next({"offer_escalation": True}), "retrieve")
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

    def test_offer_includes_a_human_option(self):
        values = " ".join(option["value"] for option in graph.SOFT_ESCALATION_OPTIONS)
        self.assertIn("human agent", values)


class EscalationAckTest(unittest.TestCase):
    def test_ack_mentions_a_colleague(self):
        ack = graph._escalation_ack({"intent": "human_request"})
        self.assertIn("human agent", ack)

    def test_safety_ack_points_at_emergency_services(self):
        ack = graph._escalation_ack({"intent": "safety"})
        self.assertIn("emergency", ack.lower())


class GreetingTest(unittest.IsolatedAsyncioTestCase):
    async def test_greeting_short_circuits_the_turn(self):
        result = await graph.dispatch({"query": "Hi"})
        self.assertEqual(result["greeting"], "greeting")
        self.assertIn("hermes", result["answer"])
        self.assertTrue(result["escalation_offer"]["options"])

    async def test_real_question_is_not_a_greeting(self):
        result = await graph.dispatch({"query": "Hi, I need to rebook my flight"})
        self.assertNotIn("greeting", result)


if __name__ == "__main__":
    unittest.main()
