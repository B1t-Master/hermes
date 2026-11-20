"""WebSocket endpoint tests (passenger + dashboard channels).

External seams are patched at the ``app.api.ws`` module boundary: token
decoding, conversation lookup, the DB session maker, the turn pipeline and
turn notifications. The in-memory hubs are used for real so room fan-out is
exercised end-to-end.
"""

import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import app
from app.models import (
    Agent,
    ConversationStatus,
    Escalation,
    EscalationStatus,
)
from tests.support.ws_fakes import FakeDB, FakeSessionMaker

WS = "app.api.ws"


def _conversation(passenger_id, status=ConversationStatus.ACTIVE):
    return SimpleNamespace(
        id=uuid.uuid4(), passenger_id=passenger_id, status=status
    )


class _WsTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.passenger_id = uuid.uuid4()
        self.agent_id = uuid.uuid4()
        self.conversation = _conversation(self.passenger_id)
        self._start(
            patch(f"{WS}.get_conversation", new=AsyncMock(return_value=self.conversation)),
            patch(f"{WS}.SessionLocal", new=MagicMock()),
        )

    def _start(self, *patchers):
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    # -- helpers ---------------------------------------------------------

    def _refused(self, auth, url, expect_code):
        """Connect and assert the socket is refused with a close code."""
        self._start(patch(f"{WS}.decode_access_token", **auth))
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self.client.websocket_connect(url) as ws:
                ws.receive_json()
        self.assertEqual(ctx.exception.code, expect_code)


class ConversationSocketAuthTest(_WsTest):
    def test_invalid_token_close_4401(self):
        self._refused(
            {"side_effect": jwt.InvalidTokenError("bad")},
            f"/ws/conversations/{self.conversation.id}?token=x",
            4401,
        )

    def test_unknown_role_close_4401(self):
        self._refused(
            {"return_value": {"role": "admin", "sub": str(self.passenger_id)}},
            f"/ws/conversations/{self.conversation.id}?token=x",
            4401,
        )

    def test_malformed_sub_close_4401(self):
        self._refused(
            {"return_value": {"role": "passenger", "sub": "not-a-uuid"}},
            f"/ws/conversations/{self.conversation.id}?token=x",
            4401,
        )

    def test_unknown_conversation_close_4403(self):
        self._start(patch(f"{WS}.get_conversation", new=AsyncMock(return_value=None)))
        self._refused(
            {"return_value": {"role": "passenger", "sub": str(self.passenger_id)}},
            f"/ws/conversations/{uuid.uuid4()}?token=x",
            4403,
        )

    def test_passenger_not_owner_close_4403(self):
        self._refused(
            {"return_value": {"role": "passenger", "sub": str(uuid.uuid4())}},
            f"/ws/conversations/{self.conversation.id}?token=x",
            4403,
        )

    def test_resolved_conversation_close_4410(self):
        self.conversation.status = ConversationStatus.RESOLVED
        self._refused(
            {"return_value": {"role": "passenger", "sub": str(self.passenger_id)}},
            f"/ws/conversations/{self.conversation.id}?token=x",
            4410,
        )

    def test_agent_must_have_claimed_the_escalation(self):
        self._start(
            patch(f"{WS}.decode_access_token", return_value={
                "role": "agent", "sub": str(self.agent_id)
            }),
            patch(f"{WS}.SessionLocal", new=FakeSessionMaker(FakeDB())),
        )
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self.client.websocket_connect(
                f"/ws/conversations/{self.conversation.id}?token=x"
            ) as ws:
                ws.receive_json()
        self.assertEqual(ctx.exception.code, 4403)


class PassengerTurnTest(_WsTest):
    def _connect(self):
        """Start patches, return an unentered websocket session."""
        self._start(
            patch(f"{WS}.decode_access_token", return_value={
                "role": "passenger", "sub": str(self.passenger_id)
            }),
            patch(f"{WS}.notify_turn", new=AsyncMock()),
        )
        return self.client.websocket_connect(
            f"/ws/conversations/{self.conversation.id}?token=x"
        )

    def _enter(self, ws):
        session = ws.__enter__()
        self.addCleanup(ws.__exit__, None, None, None)
        return session

    def test_passenger_turn_streams_tokens_and_final_message(self):
        async def fake_run(cid, content, on_token=None):
            if on_token:
                await on_token("Hello ")
                await on_token("world")
            return {
                "passenger_message": {"id": "p1", "content": content},
                "bot_message": {"id": "b1", "content": "Hello world"},
            }

        self._start(patch(f"{WS}.run_turn", side_effect=fake_run))
        with self._connect() as ws:
            ws.send_json({"type": "user_message", "content": "hi"})
            self.assertEqual(ws.receive_json(), {"type": "typing", "sender": "bot"})
            self.assertEqual(ws.receive_json(), {"type": "bot_token", "content": "Hello "})
            self.assertEqual(ws.receive_json(), {"type": "bot_token", "content": "world"})
            done = ws.receive_json()
            self.assertEqual(done["type"], "bot_done")
            self.assertEqual(done["message"]["content"], "Hello world")

    def test_empty_and_overlong_messages_are_rejected(self):
        with self._connect() as ws:
            ws.send_json({"type": "user_message", "content": "   "})
            self.assertEqual(ws.receive_json()["detail"], "empty message")
            ws.send_json({"type": "user_message", "content": "x" * 4001})
            self.assertEqual(ws.receive_json()["detail"], "message too long")

    def test_permission_error_maps_to_bot_error(self):
        self._start(patch(f"{WS}.run_turn", new=AsyncMock(side_effect=PermissionError("closed"))))
        with self._connect() as ws:
            ws.send_json({"type": "user_message", "content": "hi"})
            self.assertEqual(ws.receive_json()["type"], "typing")
            err = ws.receive_json()
            self.assertEqual(err["type"], "bot_error")
            self.assertEqual(err["detail"], "closed")

    def test_unexpected_turn_error_sends_generic_bot_error(self):
        self._start(patch(f"{WS}.run_turn", new=AsyncMock(side_effect=RuntimeError("boom"))))
        with self._connect() as ws:
            ws.send_json({"type": "user_message", "content": "hi"})
            self.assertEqual(ws.receive_json()["type"], "typing")
            err = ws.receive_json()
            self.assertEqual(err["type"], "bot_error")
            self.assertIn("couldn't process", err["detail"])

    def test_escalated_turn_without_bot_message_still_finishes(self):
        self._start(patch(f"{WS}.run_turn", new=AsyncMock(return_value={"escalate": True})))
        with self._connect() as ws:
            ws.send_json({"type": "user_message", "content": "hi"})
            self.assertEqual(ws.receive_json()["type"], "typing")
            done = ws.receive_json()
            self.assertEqual(done["type"], "bot_done")
            self.assertIsNone(done["message"])

    def test_turn_without_answer_is_a_bot_error(self):
        self._start(patch(f"{WS}.run_turn", new=AsyncMock(return_value={})))
        with self._connect() as ws:
            ws.send_json({"type": "user_message", "content": "hi"})
            self.assertEqual(ws.receive_json()["type"], "typing")
            err = ws.receive_json()
            self.assertEqual(err["type"], "bot_error")
            self.assertEqual(err["detail"], "no answer generated")

    def test_ping_pong_typing_and_unsupported_frames(self):
        with self._connect() as ws:
            ws.send_json({"type": "ping"})
            self.assertEqual(ws.receive_json(), {"type": "pong"})
            ws.send_json({"type": "pong"})
            ws.send_json({"type": "typing"})
            typing = ws.receive_json()
            self.assertEqual(typing["type"], "typing")
            self.assertEqual(typing["sender"], "passenger")
            ws.send_json({"type": "agent_message", "content": "hi"})
            err = ws.receive_json()
            self.assertEqual(err["type"], "error")
            self.assertIn("agent_message", err["detail"])


class AgentSocketTest(_WsTest):
    def _agent_connect(self, db):
        self._start(
            patch(f"{WS}.decode_access_token", return_value={
                "role": "agent", "sub": str(self.agent_id)
            }),
            patch(f"{WS}.SessionLocal", new=FakeSessionMaker(db)),
            patch(f"{WS}.message_to_dict", lambda m: {"id": "a1", "content": m.content}),
        )
        return self.client.websocket_connect(
            f"/ws/conversations/{self.conversation.id}?token=x"
        )

    def _claimed_db(self):
        db = FakeDB()
        db.register(Escalation(
            conversation_id=self.conversation.id,
            agent_id=self.agent_id,
            status=EscalationStatus.ASSIGNED,
        ))
        return db

    def test_agent_message_is_persisted_and_broadcast(self):
        with self._agent_connect(self._claimed_db()) as ws:
            ws.send_json({"type": "agent_message", "content": "Hello from the agent"})
            event = ws.receive_json()
            self.assertEqual(event["type"], "agent_message")
            self.assertEqual(event["message"]["content"], "Hello from the agent")

    def test_released_claim_closes_the_socket(self):
        db = self._claimed_db()
        with self._agent_connect(db) as ws:
            ws.send_json({"type": "agent_message", "content": "hello?"})
            err = ws.receive_json()
            self.assertEqual(err["type"], "agent_message")
            # The claim is released between messages.
            db.store[Escalation].clear()
            ws.send_json({"type": "agent_message", "content": "still there?"})
            err = ws.receive_json()
            self.assertEqual(err["type"], "error")
            self.assertIn("no longer own", err["detail"])
            with self.assertRaises(WebSocketDisconnect) as ctx:
                ws.receive_json()
            self.assertEqual(ctx.exception.code, 4403)

    def test_blank_agent_message_is_rejected(self):
        with self._agent_connect(self._claimed_db()) as ws:
            ws.send_json({"type": "agent_message", "content": "  "})
            err = ws.receive_json()
            self.assertEqual(err["type"], "error")
            self.assertEqual(err["detail"], "invalid message")


class DashboardSocketTest(_WsTest):
    def test_non_agent_token_is_refused(self):
        self._refused(
            {"return_value": {"role": "passenger", "sub": str(self.passenger_id)}},
            "/ws/dashboard?token=x",
            4401,
        )

    def test_unknown_agent_is_refused(self):
        self._start(
            patch(f"{WS}.decode_access_token", return_value={
                "role": "agent", "sub": str(self.agent_id)
            }),
            patch(f"{WS}.SessionLocal", new=FakeSessionMaker(FakeDB())),
        )
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self.client.websocket_connect("/ws/dashboard?token=x") as ws:
                ws.receive_json()
        self.assertEqual(ctx.exception.code, 4401)

    def test_dashboard_receives_heartbeats(self):
        db = FakeDB()
        db.store[Agent] = [SimpleNamespace(id=self.agent_id)]
        self._start(
            patch(f"{WS}.decode_access_token", return_value={
                "role": "agent", "sub": str(self.agent_id)
            }),
            patch(f"{WS}.SessionLocal", new=FakeSessionMaker(db)),
        )
        with self.client.websocket_connect("/ws/dashboard?token=x") as ws:
            ws.send_json({"type": "ping"})
            self.assertEqual(ws.receive_json(), {"type": "pong"})


if __name__ == "__main__":
    unittest.main()
