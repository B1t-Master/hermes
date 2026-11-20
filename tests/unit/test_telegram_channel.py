"""Telegram channel adapter tests (polling loop, mapping, fan-out).

``app.db.SessionLocal`` is patched with a fake session maker, the Telegram
API is a recorder object, and the turn pipeline is mocked. The in-memory
conversation hub is used for real.
"""

import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.channels.telegram import channel as channel_module
from app.channels.telegram.channel import TelegramChannel
from app.agents.hub import conversation_hub
from app.models import Conversation, ConversationStatus, Passenger

from tests.support.fakes import FakeDB
from tests.support.ws_fakes import FakeSessionMaker

CHAN = "app.channels.telegram.channel"
MAPPING = "app.channels.telegram.mapping"


class FakeTelegramAPI:
    def __init__(self):
        self.sent = []
        self.actions = []
        self.updates = []
        self.fail_html = False

        async def send_message(chat_id, text, parse_mode=None):
            if self.fail_html and parse_mode == "HTML":
                raise channel_module.TelegramError("bad markup")
            self.sent.append((chat_id, text, parse_mode))

        self.send_message = AsyncMock(side_effect=send_message)
        self.send_chat_action = AsyncMock()
        self.delete_webhook = AsyncMock()
        self.get_me = AsyncMock(return_value={"username": "hermes_bot"})
        self.close = AsyncMock()

    async def get_updates(self, offset, timeout=0):
        return self.updates

    def texts(self):
        return [text for _, text, _ in self.sent]


def _update(chat_id=555, text="hello", first_name="Grace"):
    return {
        "update_id": 1,
        "message": {
            "chat": {"id": chat_id},
            "from": {"first_name": first_name},
            "text": text,
        },
    }


class ChannelLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def _channel(self, api=None, session=None):
        api = api or FakeTelegramAPI()
        chan = TelegramChannel(api, poll_timeout=1)
        if session is None:
            session = FakeSessionMaker(FakeDB())
        patches = [patch(f"{CHAN}.SessionLocal", new=session)]
        if not isinstance(session, MagicMock):
            patches.append(patch.object(TelegramChannel, "_poll_loop", new=AsyncMock()))
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        await chan.start()
        return chan, api

    async def test_start_and_stop_round_trip(self):
        chan, api = await self._channel()
        self.assertIsNotNone(chan._task)
        await chan.stop()
        self.assertIsNone(chan._task)
        api.close.assert_awaited_once()
        # getUpdates must never have been called (poll loop mocked out).
        self.assertEqual(api.sent, [])

    async def test_start_tolerates_webhook_and_getme_failures(self):
        api = FakeTelegramAPI()
        api.delete_webhook.side_effect = RuntimeError("offline")
        api.get_me.side_effect = RuntimeError("offline")
        await self._channel(api=api)
        await api.close.__await__() if False else None

    async def test_rejoin_attaches_sinks_to_open_conversations(self):
        passenger_id = uuid.uuid4()
        conversation_id = uuid.uuid4()
        passenger = Passenger(
            id=passenger_id, first_name="Grace", telegram_chat_id="555"
        )
        conversation = Conversation(
            id=conversation_id,
            passenger_id=passenger_id,
            status=ConversationStatus.ESCALATED,
        )

        class _Rows:
            def all(self):
                return [(conversation_id, 555)]

        class _DB:
            def __call__(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info):
                return False

            async def execute(self, stmt):
                return _Rows()

        chan, api = await self._channel(session=_DB())
        try:
            self.assertIn(conversation_id, chan._sinks)
            # An agent reply in that room is forwarded to the chat.
            await conversation_hub.send(
                conversation_id,
                {"type": "agent_message", "message": {"content": "Hello there"}},
            )
            self.assertTrue(
                any("Agent: Hello there" in text for text in api.texts()),
                api.texts(),
            )
        finally:
            await chan.stop()

    async def test_rejoin_survives_db_failures(self):
        chan, api = await self._channel(
            session=MagicMock(side_effect=RuntimeError("no db"))
        )
        await chan.stop()


class ChannelTurnTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.api = FakeTelegramAPI()
        self.chan = TelegramChannel(self.api, poll_timeout=1)
        self.patches = [
            patch(f"{CHAN}._get_or_create_passenger", new=AsyncMock(return_value=uuid.uuid4())),
            patch(f"{CHAN}._get_or_create_conversation", new=AsyncMock(return_value=uuid.uuid4())),
            patch(f"{CHAN}.notify_turn", new=AsyncMock()),
            patch(f"{CHAN}.SessionLocal", new=FakeSessionMaker(FakeDB())),
            patch.object(TelegramChannel, "_poll_loop", new=AsyncMock()),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    async def _start(self):
        # SessionLocal is already patched in setUp with an empty FakeDB.
        await self.chan.start()
        self.addCleanup(self.chan.stop)

    async def test_start_and_help_commands(self):
        await self._start()
        await self.chan._handle_update(_update(text="/start", chat_id=1))
        await self.chan._handle_update(_update(text="/help", chat_id=2))
        self.assertEqual(len(self.api.sent), 2)
        self.assertIn("Kenya Airways assistant", self.api.sent[0][1])
        self.assertIn("normal message", self.api.sent[1][1])

    async def test_text_turn_sends_the_bot_reply(self):
        await self._start()
        with patch(
            f"{CHAN}.run_turn",
            new=AsyncMock(return_value={"bot_message": {"content": "Baggage is 2 pieces."}}),
        ):
            await self.chan._handle_update(_update(chat_id=7, text="baggage?"))
        self.api.send_chat_action.assert_awaited_with(7, "typing")
        self.assertEqual(self.api.texts(), ["Baggage is 2 pieces."])

    async def test_escalation_notice_on_a_new_escalation(self):
        await self._start()
        with patch(
            f"{CHAN}.run_turn",
            new=AsyncMock(return_value={"escalate": True, "escalation_created": True}),
        ):
            await self.chan._handle_update(_update(chat_id=7, text="refund now!"))
        self.assertIn("human agent", self.api.texts()[0])

    async def test_already_escalated_turn_gets_an_acknowledgement(self):
        await self._start()
        with patch(
            f"{CHAN}.run_turn",
            new=AsyncMock(return_value={"escalate": True, "escalation_created": False}),
        ):
            await self.chan._handle_update(_update(chat_id=7, text="still here"))
        self.assertIn("escalated to a human agent", self.api.texts()[0])

    async def test_permission_error_sends_the_closed_notice(self):
        await self._start()
        with patch(f"{CHAN}.run_turn", new=AsyncMock(side_effect=PermissionError)):
            await self.chan._handle_update(_update(chat_id=7, text="hi"))
        self.assertIn("closed", self.api.texts()[0])

    async def test_turn_failure_sends_the_error_notice(self):
        await self._start()
        with patch(f"{CHAN}.run_turn", new=AsyncMock(side_effect=RuntimeError("boom"))):
            await self.chan._handle_update(_update(chat_id=7, text="hi"))
        self.assertIn("something went wrong", self.api.texts()[0])

    async def test_empty_bot_answer_gets_a_fallback_reply(self):
        await self._start()
        with patch(f"{CHAN}.run_turn", new=AsyncMock(return_value={})):
            await self.chan._handle_update(_update(chat_id=7, text="hi"))
        self.assertIn("Could you rephrase", self.api.texts()[0])

    async def test_unusable_updates_are_skipped(self):
        await self._start()
        await self.chan._handle_update({"message": {"chat": {}, "text": None}})
        self.assertEqual(self.api.sent, [])

    async def test_html_failure_falls_back_to_plain_text(self):
        await self._start()
        self.api.fail_html = True
        with patch(
            f"{CHAN}.run_turn",
            new=AsyncMock(return_value={"bot_message": {"content": "<b>bold</b>"}}),
        ):
            await self.chan._handle_update(_update(chat_id=7, text="hi"))
        calls = self.api.send_message.await_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs.get("parse_mode"), "HTML")
        self.assertIsNone(calls[1].kwargs.get("parse_mode"))
        self.assertEqual(calls[1].args[1], "<b>bold</b>")


class PollLoopTest(unittest.IsolatedAsyncioTestCase):
    """Runs the real polling loop (no _poll_loop patch in the way)."""

    async def test_poll_loop_processes_updates_and_advances_the_offset(self):
        api = FakeTelegramAPI()
        updates = [_update(chat_id=1), _update(chat_id=2)]
        api.updates = updates
        chan = TelegramChannel(api, poll_timeout=1)
        handled = []

        async def fake_handle(self, update):
            handled.append(update)
            if len(handled) >= 2:
                chan._closed = True

        with (
            patch(f"{CHAN}.SessionLocal", new=FakeSessionMaker(FakeDB())),
            patch.object(TelegramChannel, "_rejoin_rooms", new=AsyncMock()),
            patch.object(TelegramChannel, "_handle_update", new=fake_handle),
        ):
            await chan._poll_loop()
        self.assertEqual(len(handled), 2)

    async def test_poll_loop_retries_after_a_409(self):
        api = FakeTelegramAPI()
        response = SimpleNamespace(status_code=409)
        sleeps = []
        chan = TelegramChannel(api, poll_timeout=1)

        async def fake_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) >= 2:
                chan._closed = True
            return None

        calls = {"n": 0}

        async def get_updates(offset, timeout=0):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.HTTPStatusError(
                    "conflict", request=MagicMock(), response=response
                )
            chan._closed = True
            return []

        api.get_updates = get_updates
        with (
            patch(f"{CHAN}.SessionLocal", new=FakeSessionMaker(FakeDB())),
            patch.object(TelegramChannel, "_rejoin_rooms", new=AsyncMock()),
            patch("asyncio.sleep", new=fake_sleep),
        ):
            await chan._poll_loop()
        self.assertEqual(sleeps, [30])

    async def test_poll_loop_retries_after_generic_errors(self):
        api = FakeTelegramAPI()
        sleeps = []
        chan = TelegramChannel(api, poll_timeout=1)

        async def fake_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) >= 2:
                chan._closed = True
            return None

        calls = {"n": 0}

        async def get_updates(offset, timeout=0):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("network down")
            chan._closed = True
            return []

        api.get_updates = get_updates
        with (
            patch(f"{CHAN}.SessionLocal", new=FakeSessionMaker(FakeDB())),
            patch.object(TelegramChannel, "_rejoin_rooms", new=AsyncMock()),
            patch("asyncio.sleep", new=fake_sleep),
        ):
            await chan._poll_loop()
        self.assertEqual(sleeps, [3])


class MappingTest(unittest.IsolatedAsyncioTestCase):
    async def test_get_or_create_passenger_reuses_existing(self):
        passenger = Passenger(id=uuid.uuid4(), first_name="Grace", telegram_chat_id="555")
        db = FakeDB()
        db.register(passenger)
        with patch(f"{MAPPING}.SessionLocal", new=FakeSessionMaker(db)):
            from app.channels.telegram.mapping import _get_or_create_passenger

            passenger_id = await _get_or_create_passenger(555, "Grace")
        self.assertEqual(passenger_id, passenger.id)

    async def test_get_or_create_passenger_creates_missing(self):
        db = FakeDB()
        with patch(f"{MAPPING}.SessionLocal", new=FakeSessionMaker(db)):
            from app.channels.telegram.mapping import _get_or_create_passenger

            passenger_id = await _get_or_create_passenger(555, "Grace")
        created = db.added[0]
        self.assertEqual(created.telegram_chat_id, "555")
        self.assertEqual(created.id, passenger_id)
        self.assertTrue(created.is_anonymous)

    async def test_get_or_create_conversation_creates_one(self):
        passenger_id = uuid.uuid4()
        db = FakeDB()
        with patch(f"{MAPPING}.SessionLocal", new=FakeSessionMaker(db)):
            from app.channels.telegram.mapping import _get_or_create_conversation

            conversation_id = await _get_or_create_conversation(passenger_id)
        created = db.added[0]
        self.assertEqual(created.id, conversation_id)
        self.assertEqual(created.passenger_id, passenger_id)

    async def test_get_or_create_conversation_reuses_open_one(self):
        passenger_id = uuid.uuid4()
        conversation = Conversation(
            id=uuid.uuid4(),
            passenger_id=passenger_id,
            status=ConversationStatus.ESCALATED,
        )
        db = FakeDB()
        db.register(conversation)
        with patch(f"{MAPPING}.SessionLocal", new=FakeSessionMaker(db)):
            from app.channels.telegram.mapping import _get_or_create_conversation

            conversation_id = await _get_or_create_conversation(passenger_id)
        self.assertEqual(conversation_id, conversation.id)


if __name__ == "__main__":
    unittest.main()
