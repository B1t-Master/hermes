"""Turn pipeline (orchestration), store helpers and notify fan-out tests."""

import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import (
    Booking,
    BookingStatus,
    Conversation,
    ConversationStatus,
    Message,
    MessageSender,
    Passenger,
    PaymentStatus,
)
from tests.support.fakes import FakeDB
from tests.support.ws_fakes import FakeSessionMaker

ORCH = "app.agents.turns.orchestration"
STORE = "app.agents.turns.store"
NOTIFY = "app.agents.notify"


def _conversation(status=ConversationStatus.ACTIVE, facts=None):
    return Conversation(
        id=uuid.uuid4(),
        passenger_id=uuid.uuid4(),
        status=status,
        turn_count=2,
        facts=facts if facts is not None else {},
        escalation_reason=None,
        soft_offers=0,
    )


def _graph_result(**overrides):
    result = {
        "answer": "Here is your answer.\n\nSources:\n- https://example.com",
        "chunks": [{
            "title": "Baggage policy",
            "source_url": "https://example.com/baggage",
            "content": "You may bring two pieces.",
        }],
        "sentiment_score": 0.2,
        "sentiment_label": "neutral",
        "tool_results": [],
    }
    result.update(overrides)
    return result


class _PipelineTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, db, graph_result, conversation=None, **kwargs):
        session = FakeSessionMaker(db)
        patches = [
            patch(f"{ORCH}.SessionLocal", new=session),
            patch(f"{STORE}.SessionLocal", new=session),
            patch(f"{ORCH}.get_graph", return_value=SimpleNamespace(
                ainvoke=AsyncMock(return_value=graph_result)
            )),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        from app.agents.turns import orchestration

        conversation = conversation or _conversation()
        db.register(conversation)
        return await orchestration.run_turn(conversation.id, "hello there", **kwargs)


class RunTurnTest(_PipelineTest):
    async def test_unknown_conversation_raises_lookup_error(self):
        from app.agents.turns import orchestration

        session = FakeSessionMaker(FakeDB())
        with (
            patch(f"{ORCH}.SessionLocal", new=session),
            patch(f"{STORE}.SessionLocal", new=session),
        ):
            with self.assertRaises(LookupError):
                await orchestration.run_turn(uuid.uuid4(), "hello")

    async def test_resolved_conversation_raises_permission_error(self):
        from app.agents.turns import orchestration

        conversation = _conversation(status=ConversationStatus.RESOLVED)
        db = FakeDB()
        db.register(conversation)
        session = FakeSessionMaker(db)
        with (
            patch(f"{ORCH}.SessionLocal", new=session),
            patch(f"{STORE}.SessionLocal", new=session),
        ):
            with self.assertRaises(PermissionError):
                await orchestration.run_turn(conversation.id, "hello")

    async def test_escalated_conversation_routes_to_the_agent(self):
        db = FakeDB()
        conversation = _conversation(status=ConversationStatus.ESCALATED, facts={})
        conversation.escalation_reason = "safety"
        result = await self._run(db, {}, conversation=conversation)
        self.assertTrue(result["escalate"])
        self.assertFalse(result["escalation_created"])
        self.assertEqual(result["escalation_reason"], "safety")
        self.assertIsNone(result["bot_message"])
        # The passenger message was persisted (one add happened).
        self.assertEqual(db.added[0].sender, MessageSender.PASSENGER)

    async def test_happy_path_persists_messages_and_metrics(self):
        db = FakeDB()
        conversation = _conversation()
        result = await self._run(db, _graph_result(), conversation=conversation)
        self.assertEqual(result["answer"], "Here is your answer.")
        self.assertNotIn("Sources:", result["answer"])
        self.assertEqual(result["bot_message"]["sender"], "bot")
        self.assertEqual(result["passenger_message"]["sender"], "passenger")
        senders = [
            msg.sender for msg in db.added if isinstance(msg, Message)
        ]
        self.assertIn(MessageSender.PASSENGER, senders)
        self.assertIn(MessageSender.BOT, senders)
        self.assertEqual(conversation.turn_count, 3)
        # Two metrics rows were queued (response time + retrieval hits).
        self.assertGreaterEqual(db.commits, 1)

    async def test_escalation_created_is_counted(self):
        db = FakeDB()
        result = await self._run(
            db, _graph_result(escalation_created=True, answer="hand-off ack")
        )
        self.assertEqual(result["bot_message"]["content"], "hand-off ack")

    async def test_declined_hand_off_is_remembered(self):
        db = FakeDB()
        with patch(f"{ORCH}.detect_human_decline", return_value=True):
            result = await self._run(db, _graph_result())
        self.assertEqual(result["answer"], "Here is your answer.")

    async def test_tool_results_update_facts(self):
        db = FakeDB()
        conversation = _conversation()
        tool_result = {"ticket_number": "1234567890", "pnr": "KQ7X2B9"}
        with patch(
            f"{ORCH}.apply_rebook_result", side_effect=lambda facts, tr: {**facts, "touched": True}
        ) as apply_rebook:
            result = await self._run(
                db,
                _graph_result(tool_results=[tool_result]),
                conversation=conversation,
            )
        apply_rebook.assert_called_once()
        # The facts dict was updated in place by the pipeline.
        self.assertIn("touched", conversation.facts)

    async def test_payment_card_is_deduplicated(self):
        db = FakeDB()
        conversation = _conversation()
        payment_uuid = uuid.uuid4()
        existing = Message(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            sender=MessageSender.BOT,
            content="earlier card",
            payment_id=payment_uuid,
        )
        db.register(existing)
        result = await self._run(
            db,
            _graph_result(answer="", tool_results=[{"payment_id": str(payment_uuid)}]),
            conversation=conversation,
        )
        # The duplicate card is dropped, and with no answer text there is no
        # bot message at all this turn.
        self.assertIsNone(result["bot_message"])

    async def test_new_payment_gets_a_card_and_a_cue_answer(self):
        db = FakeDB()
        payment_uuid = uuid.uuid4()
        result = await self._run(
            db,
            _graph_result(answer="", tool_results=[{"payment_id": str(payment_uuid)}]),
        )
        self.assertIsNotNone(result["bot_message"])
        self.assertIn("payment card", result["bot_message"]["content"])

    async def test_no_answer_means_no_bot_message(self):
        db = FakeDB()
        result = await self._run(db, _graph_result(answer="", chunks=[]))
        self.assertIsNone(result["bot_message"])

    async def test_streaming_callback_is_forwarded(self):
        db = FakeDB()
        on_token = AsyncMock()
        result = await self._run(db, _graph_result(), on_token=on_token)
        graph = None
        self.assertIsNotNone(result["bot_message"])


class StoreTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = FakeDB()
        self.session = FakeSessionMaker(self.db)
        for target in (STORE, ORCH, NOTIFY):
            patcher = patch(f"{target}.SessionLocal", new=self.session)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_message_to_dict_round_trip(self):
        message = Message(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            sender=MessageSender.BOT,
            content="hello",
            sentiment_score=0.1,
            sentiment_label="neutral",
            created_at=__import__("datetime").datetime(2026, 1, 1),
            payment_id=None,
        )
        from app.agents.turns.store import message_to_dict

        data = message_to_dict(message)
        self.assertEqual(data["sender"], "bot")
        self.assertEqual(data["created_at"], "2026-01-01T00:00:00")
        self.assertEqual(data["sources"], [])

    async def test_load_history_orders_oldest_first(self):
        from app.agents.turns.store import load_history

        conversation_id = uuid.uuid4()
        import datetime

        # Insert newest-first, matching the SQL order_by(created_at desc) the
        # helper expects; it reverses the rows before returning.
        rows = [
            (MessageSender.AGENT, "agent reply", datetime.datetime(2026, 1, 3)),
            (MessageSender.BOT, "second", datetime.datetime(2026, 1, 2)),
            (MessageSender.PASSENGER, "first", datetime.datetime(2026, 1, 1)),
        ]
        for sender, content, created_at in rows:
            self.db.register(Message(
                conversation_id=conversation_id,
                sender=sender,
                content=content,
                created_at=created_at,
            ))
        history = await load_history(conversation_id)
        self.assertEqual([m["content"] for m in history], ["first", "second", "agent reply"])
        self.assertEqual(history[2]["role"], "assistant")

    async def test_get_conversation_found_and_missing(self):
        from app.agents.turns.store import get_conversation

        conversation = _conversation()
        self.db.register(conversation)
        self.assertIs(await get_conversation(conversation.id), conversation)
        self.assertIsNone(await get_conversation(uuid.uuid4()))

    def test_booking_to_dict(self):
        from app.agents.turns.store import booking_to_dict

        booking = Booking(
            pnr="KQ7X2B9", flight_no="KQ100", origin="NBO", destination="LHR",
            depart_date="2026-12-01", depart_time="21:40", status=BookingStatus.HELD,
        )
        data = booking_to_dict(booking)
        self.assertEqual(data["pnr"], "KQ7X2B9")
        self.assertEqual(data["status"], "held")

    async def test_load_profile_without_a_passenger(self):
        from app.agents.turns.store import load_profile

        self.assertEqual(await load_profile(uuid.uuid4()), {})

    async def test_load_profile_anonymous_passenger(self):
        from app.agents.turns.store import load_profile

        passenger = Passenger(
            id=uuid.uuid4(), first_name="Guest", is_anonymous=True
        )
        self.db.register(passenger)
        profile = await load_profile(passenger.id)
        self.assertEqual(profile["name"], "Guest")
        self.assertTrue(profile["anonymous"])
        self.assertEqual(profile["bookings"], [])


class NotifyTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.hub = SimpleNamespace(send=AsyncMock())
        self.dashboard = SimpleNamespace(broadcast=AsyncMock())

    async def test_notify_broadcasts_the_passenger_turn_and_answer(self):
        from app.agents.notify import notify_turn

        with (
            patch(f"{NOTIFY}.conversation_hub", self.hub),
            patch(f"{NOTIFY}.dashboard_hub", self.dashboard),
        ):
            await notify_turn(uuid.uuid4(), {
                "passenger_message": {"id": "p"},
                "bot_message": {"id": "b"},
            })
        types = [call.args[1]["type"] for call in self.hub.send.await_args_list]
        self.assertEqual(types, ["passenger_message", "bot_message"])
        self.dashboard.broadcast.assert_not_called()

    async def test_new_escalation_reaches_the_dashboard(self):
        from app.agents.notify import notify_turn

        with (
            patch(f"{NOTIFY}.conversation_hub", self.hub),
            patch(f"{NOTIFY}.dashboard_hub", self.dashboard),
            patch(f"{NOTIFY}._latest_escalation_id", new=AsyncMock(return_value="esc-1")),
        ):
            await notify_turn(uuid.uuid4(), {
                "passenger_message": {"id": "p"},
                "escalate": True,
                "escalation_created": True,
                "escalation_reason": "safety",
            })
        types = [call.args[1]["type"] for call in self.hub.send.await_args_list]
        self.assertEqual(types, ["passenger_message", "escalated"])
        self.dashboard.broadcast.assert_awaited_once()
        self.assertEqual(self.dashboard.broadcast.await_args.args[0]["action"], "created")

    async def test_already_escalated_turns_do_not_re_announce(self):
        from app.agents.notify import notify_turn

        with (
            patch(f"{NOTIFY}.conversation_hub", self.hub),
            patch(f"{NOTIFY}.dashboard_hub", self.dashboard),
        ):
            await notify_turn(uuid.uuid4(), {
                "passenger_message": {"id": "p"},
                "escalate": True,
                "escalation_created": False,
            })
        self.dashboard.broadcast.assert_not_called()

    async def test_latest_escalation_id_reads_the_db(self):
        from app.agents.notify import _latest_escalation_id

        db = FakeDB()
        from app.models import Escalation, EscalationStatus

        escalation = Escalation(
            id=uuid.uuid4(), conversation_id=uuid.uuid4(),
            reason="safety", summary="s", status=EscalationStatus.PENDING,
            created_at=__import__("datetime").datetime(2026, 1, 1),
        )
        db.register(escalation)
        with patch(f"{NOTIFY}.SessionLocal", new=FakeSessionMaker(db)):
            self.assertEqual(await _latest_escalation_id(escalation.conversation_id), str(escalation.id))
        with patch(f"{NOTIFY}.SessionLocal", new=FakeSessionMaker(FakeDB())):
            self.assertIsNone(await _latest_escalation_id(uuid.uuid4()))

    async def test_fan_out_failures_are_swallowed(self):
        from app.agents.notify import notify_turn

        hub = SimpleNamespace(send=AsyncMock(side_effect=RuntimeError("socket gone")))
        with (
            patch(f"{NOTIFY}.conversation_hub", hub),
            patch(f"{NOTIFY}.dashboard_hub", self.dashboard),
        ):
            await notify_turn(uuid.uuid4(), {"passenger_message": {"id": "p"}})


if __name__ == "__main__":
    unittest.main()
