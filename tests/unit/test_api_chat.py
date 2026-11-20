"""Chat + agent-dashboard + M-Pesa callback endpoint tests (FakeDB-backed)."""

import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app import security
from app.agents import payments
from app.deps import get_current_agent, get_current_passenger
from app.db import get_db
from app.main import app
from app.models import (
    Agent,
    Conversation,
    ConversationStatus,
    Escalation,
    EscalationStatus,
    Message,
    MessageSender,
    Payment,
    PaymentStatus,
)
from tests.support.fakes import FakeDB

CHAT = "app.api.chat"
AGENTS_PAY = "app.api.agents.payments"


def _passenger():
    return SimpleNamespace(id=uuid.uuid4(), email="p@example.com")


def _agent():
    return SimpleNamespace(id=uuid.uuid4(), name="Ada", email="a@kq.co.ke")


def _conversation(passenger_id, status=ConversationStatus.ACTIVE):
    return Conversation(
        id=uuid.uuid4(),
        passenger_id=passenger_id,
        status=status,
        turn_count=3,
        facts={},
    )


def _payment(conversation_id, status=PaymentStatus.PENDING):
    return Payment(
        id=uuid.uuid4(),
        conversation_id=conversation_id,
        pnr="KQ7X2B9",
        amount=78500.0,
        currency="KES",
        status=status,
    )


class _AuthedClientTest(unittest.TestCase):
    """TestClient with a FakeDB and an overridden passenger identity."""

    def setUp(self):
        self.client = TestClient(app)
        self.db = FakeDB()
        self.passenger = _passenger()
        app.dependency_overrides[get_db] = lambda: self.db
        app.dependency_overrides[get_current_passenger] = lambda: self.passenger
        self.addCleanup(app.dependency_overrides.clear)

    def _headers(self):
        token = security.create_access_token(str(self.passenger.id), "passenger")
        return {"Authorization": f"Bearer {token}"}


class ConversationRoutesTest(_AuthedClientTest):
    def test_list_conversations_returns_owned_rows(self):
        conversation = _conversation(self.passenger.id)
        self.db.register(conversation)
        resp = self.client.get("/chat/conversations", headers=self._headers())
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 1)
        self.assertEqual(resp.json()[0]["turn_count"], 3)

    def test_create_conversation_persists_one_row(self):
        resp = self.client.post("/chat/conversations", headers=self._headers())
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(self.db.commits, 1)

    def test_list_messages_requires_ownership(self):
        stranger_conversation = _conversation(uuid.uuid4())
        resp = self.client.get(
            f"/chat/conversations/{stranger_conversation.id}/messages",
            headers=self._headers(),
        )
        self.assertEqual(resp.status_code, 404)

    def test_send_message_runs_the_turn_pipeline(self):
        conversation = _conversation(self.passenger.id)
        self.db.register(conversation)

        def message(message_id, sender, content):
            return {
                "id": message_id,
                "conversation_id": str(conversation.id),
                "sender": sender,
                "content": content,
                "sentiment_score": 0.1,
                "sentiment_label": "neutral",
                "sources": [],
                "actions": None,
                "payment_id": None,
                "created_at": "2026-01-01T00:00:00",
            }

        turn_result = {
            "passenger_message": message(str(uuid.uuid4()), "passenger", "hi"),
            "bot_message": message(str(uuid.uuid4()), "bot", "Hello!"),
            "intent": "greeting",
            "sentiment_score": 0.1,
            "sentiment_label": "neutral",
        }
        with patch(
            f"{CHAT}.run_turn", new=AsyncMock(return_value=turn_result)
        ) as run_turn:
            resp = self.client.post(
                f"/chat/conversations/{conversation.id}/messages",
                json={"conversation_id": str(conversation.id), "content": "hi"},
                headers=self._headers(),
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["bot_message"]["content"], "Hello!")
        self.assertEqual(resp.json()["intent"], "greeting")
        run_turn.assert_awaited_once()

    def test_send_message_maps_pipeline_errors(self):
        conversation = _conversation(self.passenger.id)
        self.db.register(conversation)
        with patch(f"{CHAT}.run_turn", new=AsyncMock(side_effect=PermissionError("closed"))):
            resp = self.client.post(
                f"/chat/conversations/{conversation.id}/messages",
                json={"conversation_id": str(conversation.id), "content": "hi"},
                headers=self._headers(),
            )
        self.assertEqual(resp.status_code, 400)
        with patch(f"{CHAT}.run_turn", new=AsyncMock(side_effect=LookupError("gone"))):
            resp = self.client.post(
                f"/chat/conversations/{conversation.id}/messages",
                json={"conversation_id": str(conversation.id), "content": "hi"},
                headers=self._headers(),
            )
        self.assertEqual(resp.status_code, 404)

    def test_send_message_rejects_a_resolved_conversation(self):
        conversation = _conversation(self.passenger.id, ConversationStatus.RESOLVED)
        self.db.register(conversation)
        resp = self.client.post(
            f"/chat/conversations/{conversation.id}/messages",
            json={"conversation_id": str(conversation.id), "content": "hi"},
            headers=self._headers(),
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("closed", resp.json()["detail"])

    def test_csat_rates_a_conversation(self):
        conversation = _conversation(self.passenger.id)
        self.db.register(conversation)
        with patch(f"{CHAT}.add_metric") as add_metric:
            resp = self.client.post(
                f"/chat/conversations/{conversation.id}/csat",
                json={"rating": 5},
                headers=self._headers(),
            )
        self.assertEqual(resp.status_code, 204)
        add_metric.assert_called_once()
        self.assertEqual(self.db.commits, 1)


class PaymentRoutesTest(_AuthedClientTest):
    def setUp(self):
        super().setUp()
        self.hub = SimpleNamespace(send=AsyncMock())

    def _patches(self, payment, **extra):
        return [
            patch(f"{CHAT}.payments.get_payment_payload", new=AsyncMock(return_value={
                "id": str(payment.id), "pnr": "KQ7X2B9", "amount": 78500.0,
                "status": "pending", "created_at": "2026-01-01T00:00:00",
            })),
            patch(f"{CHAT}.payments.start_payment", new=AsyncMock(return_value={
                "id": str(payment.id), "pnr": "KQ7X2B9", "amount": 78500.0,
                "status": "pending", "created_at": "2026-01-01T00:00:00",
            })),
            patch(f"{CHAT}.payments.broadcast_payment_update", new=AsyncMock(return_value={
                "id": str(payment.id), "pnr": "KQ7X2B9", "amount": 78500.0,
                "status": "pending", "created_at": "2026-01-01T00:00:00",
            })),
            patch("app.db.SessionLocal", new=MagicMock()),
            patch("app.agents.hub.conversation_hub", self.hub),
        ]

    def test_get_payment_polls_state(self):
        conversation = _conversation(self.passenger.id)
        payment = _payment(conversation.id)
        self.db.register(conversation, payment)
        for patcher in self._patches(payment):
            patcher.start()
            self.addCleanup(patcher.stop)
        resp = self.client.get(f"/chat/payments/{payment.id}", headers=self._headers())
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["pnr"], "KQ7X2B9")

    def test_get_payment_requires_ownership(self):
        payment = _payment(uuid.uuid4())
        resp = self.client.get(f"/chat/payments/{payment.id}", headers=self._headers())
        self.assertEqual(resp.status_code, 404)

    def test_submit_phone_starts_an_stk_push(self):
        conversation = _conversation(self.passenger.id)
        payment = _payment(conversation.id)
        self.db.register(conversation, payment)
        for patcher in self._patches(payment):
            patcher.start()
            self.addCleanup(patcher.stop)
        resp = self.client.post(
            f"/chat/payments/{payment.id}/phone",
            json={"phone": "0712345678"},
            headers=self._headers(),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["pnr"], "KQ7X2B9")

    def test_submit_phone_maps_payment_errors_to_400(self):
        conversation = _conversation(self.passenger.id)
        payment = _payment(conversation.id)
        self.db.register(conversation, payment)
        for patcher in self._patches(payment):
            patcher.start()
            self.addCleanup(patcher.stop)
        with (
            patch("app.agents.payments.start_payment",
                  new=AsyncMock(side_effect=payments.PaymentError("bad number"))),
        ):
            resp = self.client.post(
                f"/chat/payments/{payment.id}/phone",
                json={"phone": "0712345678"},
                headers=self._headers(),
            )
        self.assertEqual(resp.status_code, 400)

    def test_email_endpoint_requires_a_successful_payment(self):
        conversation = _conversation(self.passenger.id)
        payment = _payment(conversation.id, PaymentStatus.PENDING)
        self.db.register(conversation, payment)
        for patcher in self._patches(payment):
            patcher.start()
            self.addCleanup(patcher.stop)
        resp = self.client.post(
            f"/chat/payments/{payment.id}/email",
            json={"email": "me@example.com"},
            headers=self._headers(),
        )
        self.assertEqual(resp.status_code, 409)

    def test_email_endpoint_confirms_the_booking_and_emails_the_ticket(self):
        conversation = _conversation(self.passenger.id)
        payment = _payment(conversation.id, PaymentStatus.SUCCESS)
        payment.conversation_id = conversation.id
        self.db.register(conversation, payment)
        for patcher in self._patches(payment):
            patcher.start()
            self.addCleanup(patcher.stop)

        booking_result = {
            "updated": True,
            "origin": "NBO",
            "destination": "LHR",
            "depart_date": "2026-12-01",
            "depart_time": "21:40",
            "flight_no": "KQ100",
            "amount": 78500,
            "currency": "KES",
        }
        ticket_result = {
            "ticketed": True,
            "eticket_number": "0123456789",
            "email_sent": True,
            "email_note": "",
        }

        class _FakeMsgDB:
            def __init__(self):
                self.added = []

            def __call__(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info):
                return False

            def add(self, obj):
                self.added.append(obj)

            async def commit(self):
                import datetime

                for obj in self.added:
                    if getattr(obj, "id", None) is None:
                        obj.id = uuid.uuid4()
                    if getattr(obj, "created_at", None) is None:
                        obj.created_at = datetime.datetime(2026, 1, 1)

            async def refresh(self, obj):
                return None

        fake_session = _FakeMsgDB()
        with (
            patch("services.booking.set_booking_email", new=AsyncMock(return_value=booking_result)),
            patch("services.booking.issue_ticket", new=AsyncMock(return_value=ticket_result)),
            patch("app.db.SessionLocal", new=MagicMock(return_value=fake_session)),
        ):
            resp = self.client.post(
                f"/chat/payments/{payment.id}/email",
                json={"email": "me@example.com"},
                headers=self._headers(),
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "pending")
        self.assertIn("e-ticket", str(self.hub.send.await_args_list))


class AgentDashboardTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.db = FakeDB()
        self.agent = _agent()
        app.dependency_overrides[get_db] = lambda: self.db
        app.dependency_overrides[get_current_agent] = lambda: self.agent
        self.addCleanup(app.dependency_overrides.clear)

    def _escalation(self, **overrides):
        import datetime

        fields = dict(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            reason="safety",
            summary="upset passenger",
            status=EscalationStatus.PENDING,
            created_at=datetime.datetime(2026, 1, 1),
        )
        fields.update(overrides)
        return Escalation(**fields)

    def test_escalation_messages_and_404(self):
        escalation = self._escalation(status=EscalationStatus.PENDING)
        message = Message(
            id=uuid.uuid4(),
            conversation_id=escalation.conversation_id,
            sender=MessageSender.BOT,
            content="I can help with that.",
            sentiment_score=0.1,
            sentiment_label="neutral",
            created_at=__import__("datetime").datetime(2026, 1, 1),
        )
        self.db.register(escalation, message)
        resp = self.client.get(f"/agents/escalations/{escalation.id}/messages")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()[0]["content"], "I can help with that.")
        resp = self.client.get(f"/agents/escalations/{uuid.uuid4()}/messages")
        self.assertEqual(resp.status_code, 404)

    def test_claim_takes_a_pending_escalation(self):
        escalation = self._escalation(status=EscalationStatus.PENDING)
        self.db.register(escalation)
        with patch("app.api.agents.queue._broadcast", new=AsyncMock()) as broadcast:
            resp = self.client.post(f"/agents/escalations/{escalation.id}/claim")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(escalation.status, EscalationStatus.ASSIGNED)
        self.assertEqual(escalation.agent_id, self.agent.id)
        broadcast.assert_awaited_once()

    def test_claim_conflict_when_not_pending(self):
        escalation = self._escalation(status=EscalationStatus.RESOLVED)
        self.db.register(escalation)
        resp = self.client.post(f"/agents/escalations/{escalation.id}/claim")
        self.assertEqual(resp.status_code, 409)

    def test_release_hands_the_escalation_back(self):
        escalation = self._escalation(status=EscalationStatus.ASSIGNED, agent_id=self.agent.id)
        self.db.register(escalation)
        with patch("app.api.agents.queue._broadcast", new=AsyncMock()):
            resp = self.client.post(f"/agents/escalations/{escalation.id}/release")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(escalation.status, EscalationStatus.PENDING)
        self.assertIsNone(escalation.agent_id)

    def test_release_rejects_someone_elses_escalation(self):
        escalation = self._escalation(status=EscalationStatus.ASSIGNED, agent_id=uuid.uuid4())
        self.db.register(escalation)
        resp = self.client.post(f"/agents/escalations/{escalation.id}/release")
        self.assertEqual(resp.status_code, 409)

    def test_resolve_closes_the_conversation_too(self):
        escalation = self._escalation(status=EscalationStatus.ASSIGNED, agent_id=self.agent.id)
        conversation = Conversation(
            id=escalation.conversation_id,
            status=ConversationStatus.ESCALATED,
        )
        conversation.facts = {}
        self.db.register(escalation, conversation)
        with (
            patch("app.api.agents.queue._broadcast", new=AsyncMock()),
            patch("app.agents.hub.conversation_hub.send", new=AsyncMock()),
        ):
            resp = self.client.post(f"/agents/escalations/{escalation.id}/resolve")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(escalation.status, EscalationStatus.RESOLVED)
        self.assertEqual(conversation.status, ConversationStatus.RESOLVED)

    def test_resolve_requires_the_claim(self):
        escalation = self._escalation(status=EscalationStatus.PENDING)
        self.db.register(escalation)
        resp = self.client.post(f"/agents/escalations/{escalation.id}/resolve")
        self.assertEqual(resp.status_code, 409)

    def test_dashboard_payment_view_and_send_ticket(self):
        payment = _payment(uuid.uuid4(), PaymentStatus.SUCCESS)
        self.db.register(payment)
        payload = {
            "id": str(payment.id), "pnr": "KQ7X2B9", "amount": 1.0,
            "status": "success", "created_at": "2026-01-01T00:00:00",
        }
        with (
            patch(f"{AGENTS_PAY}.payments.get_payment_payload", new=AsyncMock(return_value=payload)),
            patch(f"{AGENTS_PAY}.payments.broadcast_payment_update", new=AsyncMock()),
        ):
            resp = self.client.get(f"/agents/payments/{payment.id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["pnr"], "KQ7X2B9")

        ticket = {"ticketed": True, "eticket_number": "0123456789", "email_sent": True}
        escalation = self._escalation(
            conversation_id=payment.conversation_id,
            agent_id=self.agent.id,
            status=EscalationStatus.ASSIGNED,
        )
        self.db.register(escalation)
        with (
            patch(f"{AGENTS_PAY}.payments.get_payment_payload", new=AsyncMock(return_value=payload)),
            patch(f"{AGENTS_PAY}.payments.broadcast_payment_update", new=AsyncMock()),
            patch("services.booking.issue_ticket", new=AsyncMock(return_value=ticket)),
            patch("services.booking.ticket_message", lambda t: "Ticket sent!"),
            patch(f"{AGENTS_PAY}.conversation_hub.send", new=AsyncMock()),
            patch(f"{AGENTS_PAY}.message_to_dict", lambda m: {"id": "x"}),
        ):
            resp = self.client.post(f"/agents/payments/{payment.id}/send-ticket")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["ticket"]["eticket_number"], "0123456789")

    def test_send_ticket_requires_an_escalation_claim(self):
        payment = _payment(uuid.uuid4(), PaymentStatus.SUCCESS)
        self.db.register(payment)
        resp = self.client.post(f"/agents/payments/{payment.id}/send-ticket")
        self.assertEqual(resp.status_code, 409)

    def test_metrics_aggregates_the_rows(self):
        from app import metrics as metrics_module
        from app.models import PerformanceMetric

        row = PerformanceMetric(
            conversation_id=uuid.uuid4(),
            metric=metrics_module.RESPONSE_TIME_MS,
            value=120.0,
        )
        row.csat_rating = None
        self.db.register(row)
        resp = self.client.get("/agents/metrics")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["avg_response_ms"], 120.0)


class MpesaCallbackTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _callback(self, payload, token=None):
        url = "/daraja/callback"
        if token:
            url += f"?token={token}"
        return self.client.post(url, json=payload)

    def test_successful_callback_resolves_the_payment(self):
        body = {
            "Body": {
                "stkCallback": {
                    "CheckoutRequestID": "CO-1",
                    "ResultCode": 0,
                    "ResultDesc": "The service request is processed successfully.",
                    "CallbackMetadata": {
                        "Item": [
                            {"Name": "MpesaReceiptNumber", "Value": "R1"},
                            {"Name": "Amount", "Value": 1.0},
                        ]
                    },
                }
            }
        }
        with patch(
            "app.api.mpesa.payments.apply_payment_result_by_checkout", new=AsyncMock()
        ) as apply_result:
            resp = self._callback(body)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ResultCode": 0, "ResultDesc": "Accepted"})
        apply_result.assert_awaited_once()
        kwargs = apply_result.await_args.kwargs
        self.assertEqual(kwargs["status"], "success")
        self.assertEqual(kwargs["receipt"], "R1")

    def test_pending_callback_is_acknowledged_without_applying(self):
        body = {"Body": {"stkCallback": {
            "CheckoutRequestID": "CO-1", "ResultCode": 1037,
        }}}
        with patch(
            "app.api.mpesa.payments.apply_payment_result_by_checkout", new=AsyncMock()
        ) as apply_result:
            resp = self._callback(body)
        self.assertEqual(resp.status_code, 200)
        apply_result.assert_not_called()

    def test_missing_checkout_id_is_acknowledged(self):
        resp = self._callback({"Body": {"stkCallback": {"ResultCode": 0}}})
        self.assertEqual(resp.status_code, 200)

    def test_non_object_payload_is_acknowledged(self):
        resp = self.client.post("/daraja/callback", json=[1, 2, 3])
        self.assertEqual(resp.status_code, 200)

    def test_wrong_token_is_rejected_when_a_secret_is_set(self):
        from app.config import settings

        with patch.object(settings, "mpesa_callback_secret", "sekret"):
            resp = self._callback({}, token="wrong")
            self.assertEqual(resp.status_code, 403)
            resp = self._callback({}, token="sekret")
            self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
