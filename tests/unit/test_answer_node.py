"""Answer-generation node tests.

The node's external seams are patched: the retrieval context formatter, the
LLM client, the tool-turn runner, booking detection, the booking service's
DB-writing functions and the payment broadcaster. Parsing helpers
(``parse_route``, ``parse_travel_date``, ``_schedule_row``, ``_fare``) are
used for real — they are deterministic.
"""

import importlib
import unittest
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

answer_module = importlib.import_module("app.agents.graph.nodes.answer")
from app.agents.graph.nodes.answer import (
    _booking_context,
    _reask,
    answer,
)
from app.agents.llm import LLMError

ANS = "app.agents.graph.nodes.answer"


class FakeLLM:
    def __init__(self, text="Here is your answer.", stream_tokens=None, error=False):
        self.text = text
        self.tokens = stream_tokens or ["Here ", "is ", "your ", "answer."]
        self.error = error
        self.complete_calls = []
        self.stream_calls = []

    async def complete(self, messages, max_tokens=None, **kwargs):
        self.complete_calls.append((messages, max_tokens))
        if self.error:
            raise LLMError("provider down")
        return self.text

    async def stream(self, messages, max_tokens=None, **kwargs):
        self.stream_calls.append((messages, max_tokens))
        if self.error:
            raise LLMError("provider down")
        for token in self.tokens:
            yield token


class _AnswerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._patches = [
            patch(f"{ANS}.get_llm", return_value=FakeLLM()),
            patch(f"{ANS}._format_context", return_value=("Some context.", [])),
            patch(f"{ANS}.run_tool_turn", new=AsyncMock(return_value="")),
            patch(f"{ANS}.detect_rebook_request", return_value=False),
            patch(f"{ANS}.detect_booking_request", return_value=False),
            patch(f"{ANS}.payments.broadcast_payment_update", new=AsyncMock()),
            patch(f"{ANS}.payments.start_payment", new=AsyncMock()),
            patch("services.booking.create_booking", new=AsyncMock()),
            patch("services.booking.set_booking_email", new=AsyncMock()),
            patch("services.booking.issue_ticket", new=AsyncMock()),
            patch("services.booking.search_flights", new=MagicMock()),
        ]
        for patcher in self._patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.booking_ctx = {
            "passenger_id": "pid-1",
            "passenger_name": "Test",
            "passenger_email": "t@example.com",
            "passenger_phone": None,
            "conversation_id": "conv-1",
        }

    def _state(self, **overrides):
        state = {"query": "What is the baggage allowance?", "chunks": ["c1"]}
        state.update(overrides)
        return state

    # -- email capture ---------------------------------------------------

    async def test_email_capture_reasks_without_an_email(self):
        state = self._state(facts={"needs_email": True}, query="hmm not sure")
        result = await answer(state)
        self.assertIn("What's your email address?", result["answer"])
        self.assertEqual(result["chunks"], [])

    async def test_email_capture_without_a_pending_booking(self):
        state = self._state(facts={"needs_email": True}, query="me@example.com")
        result = await answer(state)
        self.assertIn("don't have a booking", result["answer"])

    async def test_email_saved_but_payment_pending(self):
        state = self._state(
            facts={"needs_email": True, "pending_email_booking": {"pnr": "KQ7X2B9"}},
            query="me@example.com",
        )
        from services import booking as booking_svc

        booking_svc.set_booking_email.return_value = {"updated": True, "origin": "NBO"}
        booking_svc.issue_ticket.return_value = {
            "ticketed": False,
            "email_note": "",
            "eticket_number": "",
        }
        result = await answer(state)
        self.assertIn("has been saved to booking KQ7X2B9", result["answer"])
        self.assertIn("once the M-Pesa payment completes", result["answer"])
        self.assertNotIn("needs_email", state["facts"])

    async def test_email_saved_ticketed_renders_itinerary(self):
        state = self._state(
            facts={
                "needs_email": True,
                "pending_email_booking": {"pnr": "KQ7X2B9", "payment_id": "pay-1"},
            },
            query="Me@Example.com",
        )
        from services import booking as booking_svc

        booking_svc.set_booking_email.return_value = {
            "updated": True,
            "origin": "NBO",
            "destination": "LHR",
            "date": "2026-12-01",
            "flight_no": "KQ100",
            "depart_time": "21:40",
            "amount": 78500,
            "currency": "KES",
        }
        booking_svc.issue_ticket.return_value = {
            "ticketed": True,
            "eticket_number": "0123456789",
            "email_sent": True,
            "email_note": "",
        }
        result = await answer(state)
        self.assertIn("E-ticket 0123456789 has been issued", result["answer"])
        self.assertIn("Confirmation email sent to me@example.com", result["answer"])
        self.assertIn("KQ100", result["answer"])
        self.assertIn("78,500", result["answer"])
        answer_module.payments.broadcast_payment_update.assert_awaited_with("pay-1")

    async def test_email_save_failure_returns_error(self):
        state = self._state(
            facts={"needs_email": True, "pending_email_booking": {"pnr": "KQ7X2B9"}},
            query="me@example.com",
        )
        from services import booking as booking_svc

        booking_svc.set_booking_email.return_value = {
            "updated": False,
            "error": "Booking not found",
        }
        result = await answer(state)
        self.assertEqual(result["answer"], "Booking not found")

    # -- QA / LLM path ---------------------------------------------------

    async def test_plain_answer_uses_the_llm_and_offers_escalation(self):
        state = self._state(offer_escalation=True, escalation_reason="frustration")
        result = await answer(state)
        self.assertEqual(result["answer"], "Here is your answer.")
        self.assertEqual(result["escalation_offer"]["reason"], "frustration")
        self.assertTrue(result["escalation_offer"]["options"])

    async def test_baggage_context_wins_over_the_soft_offer(self):
        state = self._state(
            baggage_context="Go to the Lost & Found desk.",
            offer_escalation=True,
            escalation_reason="frustration",
        )
        result = await answer(state)
        self.assertEqual(result["escalation_offer"]["reason"], "lost or delayed baggage")

    async def test_streaming_forwards_tokens_to_the_callback(self):
        on_token = AsyncMock()
        state = self._state()
        result = await answer(state, {"configurable": {"on_token": on_token}})
        self.assertEqual(result["answer"], "Here is your answer.")
        self.assertEqual(on_token.await_count, 4)

    async def test_llm_failure_returns_the_apology(self):
        answer_module.get_llm.return_value = FakeLLM(error=True)
        result = await answer(self._state())
        self.assertEqual(
            result["answer"],
            "I'm sorry, I couldn't process that just now. Please try again.",
        )

    async def test_rebook_tool_failure_falls_back_to_plain_qa(self):
        answer_module.detect_rebook_request.return_value = True
        answer_module.run_tool_turn.side_effect = LLMError("down")
        llm = FakeLLM(text="Plain answer.")
        answer_module.get_llm.return_value = llm
        result = await answer(self._state())
        self.assertEqual(result["answer"], "Plain answer.")
        self.assertTrue(llm.complete_calls)

    async def test_booking_request_uses_the_deterministic_fallback_first(self):
        answer_module.detect_booking_request.return_value = True
        with patch(
            f"{ANS}._booking_fallback", new=AsyncMock(return_value="Held KQ100!")
        ) as fallback:
            result = await answer(self._state())
        self.assertEqual(result["answer"], "Held KQ100!")
        fallback.assert_awaited_once()
        answer_module.run_tool_turn.assert_not_called()

    async def test_booking_tool_failure_retries_the_fallback(self):
        answer_module.detect_booking_request.return_value = True
        answer_module.run_tool_turn.side_effect = LLMError("down")
        with patch(
            f"{ANS}._booking_fallback",
            new=AsyncMock(side_effect=["", "Held KQ100!"]),
        ) as fallback:
            result = await answer(self._state())
        self.assertEqual(result["answer"], "Held KQ100!")
        self.assertEqual(fallback.await_count, 2)

    async def test_tool_results_are_attached(self):
        async def fake_tool_turn(messages, tools=None, on_token=None, results=None, context=None):
            if results is not None:
                results.append({"tool": "rebook", "ok": True})
            return "Rebooked you onto KQ101."

        answer_module.detect_rebook_request.return_value = True
        answer_module.run_tool_turn.side_effect = fake_tool_turn
        result = await answer(self._state())
        self.assertEqual(result["answer"], "Rebooked you onto KQ101.")
        self.assertEqual(result["tool_results"], [{"tool": "rebook", "ok": True}])


class BookingFallbackTest(_AnswerTest):
    """The deterministic booking path (no LLM involved)."""

    def _fallback(self, state):
        from app.agents.graph.nodes.answer import _booking_fallback

        return _booking_fallback(state, _booking_context(state), [])

    async def test_flight_and_date_without_phone_asks_for_it(self):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        # A non-empty facts dict mirrors the real pipeline, where extract_facts
        # has usually recorded something by the time booking starts.
        state = self._state(
            query=f"Book KQ100 on {tomorrow}",
            history=[],
            facts={"turns_seen": 1},
        )
        text = await self._fallback(state)
        self.assertIn("What's your phone number", text)
        self.assertIn("KQ100", text)
        self.assertEqual(
            state["facts"]["pending_booking"],
            {"flight_no": "KQ100", "travel_date": tomorrow},
        )

    async def test_pending_booking_completed_by_phone_reply(self):
        travel_date = (date.today() + timedelta(days=1)).isoformat()
        state = self._state(
            query="0712 345 678",
            facts={
                "pending_booking": {"flight_no": "KQ100", "travel_date": travel_date},
            },
            history=[],
        )
        from services import booking as booking_svc

        booking_svc.create_booking.return_value = {
            "booked": True,
            "pnr": "KQ7X2B9",
            "payment_id": None,
        }
        text = await self._fallback(state)
        self.assertIn("I've held KQ100 (NBO-LHR)", text)
        self.assertIn("Booking reference: KQ7X2B9", text)
        self.assertIn("M-Pesa payment card", text)
        self.assertNotIn("pending_booking", state["facts"])
        args = booking_svc.create_booking.await_args
        self.assertEqual(args.args[0], "KQ100")
        self.assertEqual(args.kwargs["passenger_phone"], "0712345678")

    async def test_pending_booking_triggers_the_stk_push(self):
        travel_date = (date.today() + timedelta(days=1)).isoformat()
        state = self._state(
            query="0712345678",
            facts={"pending_booking": {"flight_no": "KQ100", "travel_date": travel_date}},
            history=[],
        )
        from services import booking as booking_svc

        booking_svc.create_booking.return_value = {
            "booked": True,
            "pnr": "KQ7X2B9",
            "payment_id": "pay-9",
        }
        text = await self._fallback(state)
        self.assertIn("M-Pesa payment card", text)
        answer_module.payments.start_payment.assert_awaited_with("pay-9", "0712345678")
        answer_module.payments.broadcast_payment_update.assert_awaited_with("pay-9")

    async def test_stk_push_failure_still_reports_the_held_booking(self):
        travel_date = (date.today() + timedelta(days=1)).isoformat()
        state = self._state(
            query="0712345678",
            facts={"pending_booking": {"flight_no": "KQ100", "travel_date": travel_date}},
            history=[],
        )
        from services import booking as booking_svc

        booking_svc.create_booking.return_value = {
            "booked": True,
            "pnr": "KQ7X2B9",
            "payment_id": "pay-9",
        }
        answer_module.payments.start_payment.side_effect = RuntimeError("no credit")
        text = await self._fallback(state)
        self.assertIn("Booking reference: KQ7X2B9", text)
        self.assertIn("M-Pesa prompt could not be sent", text)

    async def test_booking_failure_returns_the_error(self):
        travel_date = (date.today() + timedelta(days=1)).isoformat()
        state = self._state(
            query="0712345678",
            facts={"pending_booking": {"flight_no": "KQ100", "travel_date": travel_date}},
            history=[],
        )
        from services import booking as booking_svc

        booking_svc.create_booking.return_value = {"booked": False, "error": "Full"}
        self.assertEqual(await self._fallback(state), "Full")

    async def test_unknown_flight_is_reported(self):
        travel_date = (date.today() + timedelta(days=1)).isoformat()
        state = self._state(
            query="0712345678",
            facts={"pending_booking": {"flight_no": "KQ999", "travel_date": travel_date}},
            history=[],
        )
        from services import booking as booking_svc

        booking_svc.create_booking.return_value = {
            "booked": True,
            "pnr": "PNR1",
            "payment_id": None,
        }
        self.assertIn("couldn't find flight KQ999", await self._fallback(state))

    async def test_repeat_phone_reask_gets_an_apology(self):
        travel_date = (date.today() + timedelta(days=1)).isoformat()
        state = self._state(
            query="hmm",
            facts={"pending_booking": {"flight_no": "KQ100", "travel_date": travel_date}},
            history=[],
        )
        first = await self._fallback(state)
        second = await self._fallback(state)
        self.assertIn("What's your phone number", first)
        self.assertTrue(second.startswith("Sorry, I didn't get that clearly."))

    async def test_flight_in_history_with_phone_creates_the_booking(self):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        state = self._state(
            query=f"Book KQ100 on {tomorrow}",
            history=[{"role": "user", "content": "hi there"}],
            facts={"phone": "0712345678"},
        )
        from services import booking as booking_svc

        booking_svc.create_booking.return_value = {
            "booked": True,
            "pnr": "KQ7X2B9",
            "payment_id": None,
        }
        text = await self._fallback(state)
        self.assertIn("I've held KQ100", text)

    async def test_stale_flight_number_in_history_does_not_book_directly(self):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        state = self._state(
            query=f"Nairobi to London on {tomorrow}",
            history=[{"role": "user", "content": "What about KQ100?"}],
        )
        from services import booking as booking_svc

        booking_svc.search_flights.return_value = {
            "available": True,
            "date": tomorrow,
            "flights": [],
        }
        text = await self._fallback(state)
        # A flight number that was not chosen in this turn triggers a search
        # for the route+date instead of a direct booking.
        self.assertIn("Flights Nairobi (NBO) to London (LHR)", text)

    async def test_route_and_date_search_lists_flights(self):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        state = self._state(
            query=f"Nairobi to London on {tomorrow}",
            history=[],
        )
        from services import booking as booking_svc

        booking_svc.search_flights.return_value = {
            "available": True,
            "date": tomorrow,
            "flights": [
                {
                    "flight_no": "KQ100",
                    "depart_time": "21:40",
                    "arrival_time": "05:15+1",
                    "fare": 78500,
                    "currency": "KES",
                    "seats_left": 9,
                }
            ],
        }
        text = await self._fallback(state)
        self.assertIn("Flights Nairobi (NBO) to London (LHR)", text)
        self.assertIn("KQ100", text)
        self.assertIn("Tell me which flight you'd like", text)

    async def test_unavailable_route_is_reported(self):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        state = self._state(
            query=f"Nairobi to London on {tomorrow}",
            history=[],
        )
        from services import booking as booking_svc

        booking_svc.search_flights.return_value = {"available": False, "flights": []}
        text = await self._fallback(state)
        self.assertIn("no available flights", text)

    async def test_route_without_date_asks_for_the_date(self):
        state = self._state(query="Nairobi to London", history=[])
        self.assertIn("Which date would you like to travel?", await self._fallback(state))

    async def test_date_without_route_asks_for_the_route(self):
        state = self._state(query="tomorrow", history=[])
        self.assertIn("Which route would you like to fly?", await self._fallback(state))

    async def test_nothing_understood_asks_for_route_and_date(self):
        state = self._state(query="help me", history=[])
        self.assertIn("Where are you flying from and to", await self._fallback(state))

    async def test_facts_new_date_is_used_when_the_query_has_none(self):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        state = self._state(
            query="Nairobi to London",
            history=[],
            facts={"new_date": tomorrow},
        )
        from services import booking as booking_svc

        booking_svc.search_flights.return_value = {
            "available": True,
            "date": tomorrow,
            "flights": [],
        }
        text = await self._fallback(state)
        self.assertIn("Flights Nairobi (NBO) to London (LHR)", text)


class SmallHelpersTest(unittest.TestCase):
    def test_reask_apologises_on_a_repeat(self):
        state = {}
        first = _reask(state, "What's your phone number?")
        second = _reask(state, "What's your phone number?")
        self.assertEqual(first, "What's your phone number?")
        self.assertTrue(second.startswith("Sorry, I didn't get that clearly."))

    def test_booking_context_collects_identity(self):
        state = {
            "passenger_id": "pid",
            "conversation_id": "cid",
            "profile": {"name": "Ada", "email": "ada@example.com"},
            "facts": {"phone": "0712", "passenger_name": "ignored"},
        }
        ctx = _booking_context(state)
        self.assertEqual(ctx["passenger_name"], "Ada")
        self.assertEqual(ctx["passenger_email"], "ada@example.com")
        self.assertEqual(ctx["passenger_phone"], "0712")

    def test_booking_context_falls_back_to_facts(self):
        ctx = _booking_context({"facts": {"passenger_name": "Grace", "email": "g@x.com"}})
        self.assertEqual(ctx["passenger_name"], "Grace")
        self.assertEqual(ctx["passenger_email"], "g@x.com")
        self.assertIsNone(ctx["passenger_id"])


if __name__ == "__main__":
    unittest.main()
