"""Payments package unit tests: state, start, polling and results.

``app.db.SessionLocal`` is patched at each module's own namespace with a
``FakeSessionMaker`` wrapping a ``FakeDB`` seeded with Payment/Booking rows.
"""

import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import (
    Booking,
    BookingStatus,
    Conversation,
    Payment,
    PaymentStatus,
)
from services import mpesa

from tests.support.fakes import FakeDB
from tests.support.ws_fakes import FakeSessionMaker

PAY_STATE = "app.agents.payments.state"
PAY_START = "app.agents.payments.start"
PAY_POLL = "app.agents.payments.polling"
PAY_RESULTS = "app.agents.payments.results"


def _payment(**overrides):
    fields = dict(
        id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        pnr="KQ7X2B9",
        phone=None,
        amount=78500.0,
        currency="KES",
        status=PaymentStatus.PENDING,
        receipt=None,
        checkout_request_id=None,
        merchant_request_id=None,
        raw=None,
    )
    fields.update(overrides)
    return Payment(**fields)


def _booking(**overrides):
    fields = dict(
        id=uuid.uuid4(),
        pnr="KQ7X2B9",
        flight_no="KQ100",
        origin="NBO",
        destination="LHR",
        depart_date="2026-12-01",
        depart_time="21:40",
        status=BookingStatus.HELD,
        passenger_email=None,
        eticket_number=None,
    )
    fields.update(overrides)
    return Booking(**fields)


class StateTest(unittest.IsolatedAsyncioTestCase):
    async def test_as_uuid_variants(self):
        from app.agents.payments.state import _as_uuid

        value = uuid.uuid4()
        self.assertEqual(_as_uuid(str(value)), value)
        self.assertEqual(_as_uuid(value), value)
        self.assertIsNone(_as_uuid(None))
        self.assertIsNone(_as_uuid("not-a-uuid"))

    def test_status_messages_cover_every_state(self):
        from app.agents.payments.state import _status_message

        payment = _payment()
        payment.checkout_request_id = None
        self.assertIn("Enter your M-Pesa number", _status_message(payment))
        payment.checkout_request_id = "CO-1"
        self.assertIn("prompt was sent", _status_message(payment))
        for status, expected in [
            (PaymentStatus.SUCCESS, "Payment received"),
            (PaymentStatus.CANCELLED, "cancelled"),
            (PaymentStatus.TIMEOUT, "timed out"),
            (PaymentStatus.FAILED, "not completed"),
        ]:
            payment.status = status
            self.assertIn(expected, _status_message(payment))

    def test_payment_payload_masks_the_phone(self):
        from app.agents.payments.state import payment_payload

        payment = _payment(phone="254712345678")
        payload = payment_payload(payment)
        self.assertEqual(payload["status"], "pending")
        self.assertNotEqual(payload["phone_masked"], "254712345678")
        self.assertEqual(payload["phone_masked"], "0712 *** 678")
        booking = _booking()
        enriched = payment_payload(payment, booking)
        self.assertEqual(enriched["flight_no"], "KQ100")
        self.assertEqual(enriched["booking_status"], "held")

    async def test_load_payment_returns_payment_and_booking(self):
        payment = _payment()
        booking = _booking()
        db = FakeDB()
        db.register(payment, booking)
        with patch(f"{PAY_STATE}.SessionLocal", new=FakeSessionMaker(db)):
            from app.agents.payments import state

            loaded_payment, loaded_booking = await state.load_payment(payment.id)
        self.assertIs(loaded_payment, payment)
        self.assertIs(loaded_booking, booking)

    async def test_load_payment_rejects_bad_ids_and_missing_rows(self):
        db = FakeDB()
        with patch(f"{PAY_STATE}.SessionLocal", new=FakeSessionMaker(db)):
            from app.agents.payments import state

            self.assertEqual(await state.load_payment("junk"), (None, None))
            self.assertEqual(await state.load_payment(uuid.uuid4()), (None, None))

    async def test_get_payment_payload_none_without_a_payment(self):
        with patch(f"{PAY_STATE}.SessionLocal", new=FakeSessionMaker(FakeDB())):
            from app.agents.payments import state

            self.assertIsNone(await state.get_payment_payload(uuid.uuid4()))

    async def test_broadcast_skips_conversations_without_a_room(self):
        from app.agents.payments import state

        hub = SimpleNamespace(send=AsyncMock())
        payment = _payment(conversation_id=None)
        with (
            patch(f"{PAY_STATE}.SessionLocal", new=FakeSessionMaker(FakeDB())),
            patch(f"{PAY_STATE}.conversation_hub", hub),
        ):
            self.assertIsNone(await state.broadcast_payment_update(payment.id))
        hub.send.assert_not_called()

    async def test_broadcast_pushes_a_payment_update_event(self):
        from app.agents.payments import state

        hub = SimpleNamespace(send=AsyncMock())
        payment = _payment()
        db = FakeDB()
        db.register(payment)
        with (
            patch(f"{PAY_STATE}.SessionLocal", new=FakeSessionMaker(db)),
            patch(f"{PAY_STATE}.conversation_hub", hub),
        ):
            payload = await state.broadcast_payment_update(payment.id)
        self.assertEqual(payload["id"], payment.id)
        hub.send.assert_awaited_once()
        event = hub.send.await_args.args[1]
        self.assertEqual(event["type"], "payment_update")


class StartPaymentTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.hub = SimpleNamespace(send=AsyncMock())
        self.poller = SimpleNamespace(schedule=AsyncMock())
        self.patches = [
            patch(f"{PAY_START}.schedule_poller", new=MagicMock()),
            patch(f"{PAY_START}._broadcast", new=AsyncMock()),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _db_with(self, *objects):
        db = FakeDB()
        db.register(*objects)
        return db

    async def test_invalid_phone_raises_payment_error(self):
        from app.agents.payments import start

        with self.assertRaises(start.PaymentError):
            await start.start_payment(uuid.uuid4(), "not-a-phone")

    async def test_unknown_payment_raises(self):
        from app.agents.payments import start

        with patch(f"{PAY_START}.SessionLocal", new=FakeSessionMaker(FakeDB())):
            with self.assertRaises(start.PaymentError):
                await start.start_payment("junk", "0712345678")
            with self.assertRaises(start.PaymentError):
                await start.start_payment(uuid.uuid4(), "0712345678")

    async def test_successful_payment_is_returned_unchanged(self):
        from app.agents.payments import start
        from services import mpesa as mpesa_svc

        payment = _payment(status=PaymentStatus.SUCCESS, phone="0712345678")
        client = SimpleNamespace(stk_push=AsyncMock())
        with (
            patch(f"{PAY_START}.SessionLocal", new=FakeSessionMaker(self._db_with(payment))),
            patch.object(mpesa_svc, "get_client", return_value=client),
        ):
            payload = await start.start_payment(payment.id, "0712345678")
        self.assertEqual(payload["status"], "success")
        client.stk_push.assert_not_called()

    async def test_stk_push_sets_checkout_ids_and_schedules_a_poller(self):
        from app.agents.payments import start
        from services import mpesa as mpesa_svc

        payment = _payment(status=PaymentStatus.CANCELLED, receipt="old")
        payment.checkout_request_id = "old-co"
        payment.merchant_request_id = "old-m"
        client = SimpleNamespace(
            stk_push=AsyncMock(return_value={
                "checkout_request_id": "CO-1",
                "merchant_request_id": "M-1",
                "raw": {"ok": True},
            })
        )
        with (
            patch(f"{PAY_START}.SessionLocal", new=FakeSessionMaker(self._db_with(payment))),
            patch.object(mpesa_svc, "get_client", return_value=client),
        ):
            payload = await start.start_payment(payment.id, "0712345678")
        self.assertEqual(payload["checkout_request_id"], "CO-1")
        self.assertEqual(payment.status, PaymentStatus.PENDING)
        self.assertIsNone(payment.receipt)
        self.assertEqual(payment.phone, "254712345678")
        start.schedule_poller.assert_called_once_with(payment.id, "CO-1")

    async def test_provider_failure_marks_the_payment_failed(self):
        from app.agents.payments import start
        from services import mpesa as mpesa_svc

        payment = _payment()
        client = SimpleNamespace(
            stk_push=AsyncMock(side_effect=mpesa_svc.MpesaError("no credit"))
        )
        with (
            patch(f"{PAY_START}.SessionLocal", new=FakeSessionMaker(self._db_with(payment))),
            patch.object(mpesa_svc, "get_client", return_value=client),
        ):
            payload = await start.start_payment(payment.id, "0712345678")
        self.assertEqual(payment.status, PaymentStatus.FAILED)
        self.assertIn("Could not send", payload["message"])
        start._broadcast.assert_awaited_once()


class PollingTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings_patches = [
            patch("app.config.settings.mpesa_stk_query_interval_seconds", 0.01),
            patch("app.config.settings.mpesa_stk_query_max_attempts", 3),
        ]
        for patcher in self.settings_patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_schedule_poller_is_idempotent_and_shutdown_cancels(self):
        from app.agents.payments import polling

        with patch(f"{PAY_POLL}._poll_payment", new=AsyncMock()):
            polling.schedule_poller("pay-1", "CO-1")
            first = polling._pollers["pay-1"]
            polling.schedule_poller("pay-1", "CO-1")
            self.assertIs(polling._pollers["pay-1"], first)
        await polling.shutdown_payments()
        self.assertEqual(polling._pollers, {})

    def test_schedule_poller_without_a_loop_is_ignored(self):
        from app.agents.payments import polling

        # No running event loop in this sync test: the RuntimeError is caught.
        polling.schedule_poller("pay-2", "CO-2")
        self.assertNotIn("pay-2", polling._pollers)

    async def test_poll_resolves_on_terminal_status(self):
        from app.agents.payments import polling

        client = SimpleNamespace(query=AsyncMock(side_effect=[
            mpesa.MpesaError("flaky"),
            {"status": "pending"},
            {"status": "success", "receipt": "R1", "result_code": "0"},
        ]))
        with (
            patch(f"{PAY_POLL}.mpesa", SimpleNamespace(
                get_client=lambda: client,
                MpesaError=mpesa.MpesaError,
                RESULT_TIMEOUT=mpesa.RESULT_TIMEOUT,
            )),
            patch(f"{PAY_POLL}.apply_payment_result", new=AsyncMock()) as apply_result,
        ):
            await polling._poll_payment("pay-1", "CO-1")
        apply_result.assert_awaited_once()
        self.assertEqual(apply_result.await_args.kwargs["status"], "success")
        self.assertEqual(apply_result.await_args.kwargs["receipt"], "R1")

    async def test_poll_times_out_when_the_budget_is_exhausted(self):
        from app.agents.payments import polling

        client = SimpleNamespace(query=AsyncMock(return_value={"status": "pending"}))
        with (
            patch(f"{PAY_POLL}.mpesa", SimpleNamespace(
                get_client=lambda: client,
                MpesaError=mpesa.MpesaError,
                RESULT_TIMEOUT=mpesa.RESULT_TIMEOUT,
            )),
            patch(f"{PAY_POLL}.apply_payment_result", new=AsyncMock()) as apply_result,
        ):
            await polling._poll_payment("pay-1", "CO-1")
        self.assertEqual(apply_result.await_args.kwargs["status"], "timeout")

    async def test_resume_schedules_pollers_for_pending_payments(self):
        from app.agents.payments import polling

        payment = _payment(checkout_request_id="CO-1")
        db = FakeDB()
        db.register(payment)
        with (
            patch(f"{PAY_POLL}.SessionLocal", new=FakeSessionMaker(db)),
            patch(f"{PAY_POLL}.schedule_poller", new=MagicMock()) as schedule,
        ):
            await polling.resume_pending_payments()
        schedule.assert_called_once_with(payment.id, "CO-1")

    async def test_resume_swallows_db_failures(self):
        from app.agents.payments import polling

        with patch(f"{PAY_POLL}.SessionLocal", new=MagicMock(side_effect=RuntimeError)):
            await polling.resume_pending_payments()


class ResultsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.hub = SimpleNamespace(send=AsyncMock())
        self.patches = [
            patch(f"{PAY_RESULTS}.conversation_hub", self.hub),
            patch(f"{PAY_STATE}.conversation_hub", self.hub),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _use_db(self, db):
        session = FakeSessionMaker(db)
        self._start = [
            patch(f"{PAY_RESULTS}.SessionLocal", new=session),
            patch(f"{PAY_STATE}.SessionLocal", new=session),
        ]
        for patcher in self._start:
            patcher.start()
            self.addCleanup(patcher.stop)
        return db

    def _db_with(self, *objects):
        db = FakeDB()
        db.register(*objects)
        return self._use_db(db)

    async def test_unknown_payment_returns_none(self):
        from app.agents.payments import results

        with patch(f"{PAY_RESULTS}.SessionLocal", new=FakeSessionMaker(FakeDB())):
            self.assertIsNone(await results.apply_payment_result(uuid.uuid4(), status="success"))

    async def test_unknown_status_is_treated_as_failed(self):
        from app.agents.payments import results

        payment = _payment()
        with patch(f"{PAY_RESULTS}.SessionLocal", new=FakeSessionMaker(self._db_with(payment))):
            payload = await results.apply_payment_result(payment.id, status="weird")
        self.assertEqual(payment.status, PaymentStatus.FAILED)
        self.assertEqual(payload["status"], "failed")

    async def test_success_marks_the_booking_paid_and_fans_out(self):
        from app.agents.payments import results

        payment = _payment()
        booking = _booking()
        conversation = Conversation(id=payment.conversation_id)
        db = FakeDB()
        db.register(payment, booking, conversation)
        self._use_db(db)
        payload = await results.apply_payment_result(
            payment.id, status="success", receipt="R9", result_code="0"
        )
        self.assertEqual(payment.status, PaymentStatus.SUCCESS)
        self.assertEqual(payment.receipt, "R9")
        self.assertEqual(booking.status, BookingStatus.PAID)
        self.assertEqual(payload["status"], "success")
        sent_types = [call.args[1]["type"] for call in self.hub.send.await_args_list]
        self.assertIn("payment_update", sent_types)
        self.assertIn("bot_message", sent_types)

    async def test_second_callback_is_idempotent(self):
        from app.agents.payments import results

        payment = _payment(status=PaymentStatus.SUCCESS)
        db = self._db_with(payment)
        with patch(f"{PAY_RESULTS}.SessionLocal", new=FakeSessionMaker(db)):
            payload = await results.apply_payment_result(payment.id, status="success")
        self.assertEqual(payload["status"], "success")
        # No bot message fanned out for the duplicate callback.
        sent_types = [call.args[1]["type"] for call in self.hub.send.await_args_list]
        self.assertNotIn("bot_message", sent_types)

    async def test_late_success_after_timeout_wins(self):
        from app.agents.payments import results

        payment = _payment(status=PaymentStatus.TIMEOUT)
        db = self._db_with(payment)
        with patch(f"{PAY_RESULTS}.SessionLocal", new=FakeSessionMaker(db)):
            payload = await results.apply_payment_result(payment.id, status="success")
        self.assertEqual(payment.status, PaymentStatus.SUCCESS)
        self.assertEqual(payload["status"], "success")

    async def test_failure_after_terminal_state_does_not_reopen(self):
        from app.agents.payments import results

        payment = _payment(status=PaymentStatus.CANCELLED)
        db = self._db_with(payment)
        with patch(f"{PAY_RESULTS}.SessionLocal", new=FakeSessionMaker(db)):
            payload = await results.apply_payment_result(payment.id, status="failed")
        self.assertEqual(payment.status, PaymentStatus.CANCELLED)
        self.assertEqual(payload["status"], "cancelled")

    async def test_callback_resolves_by_checkout_request_id(self):
        from app.agents.payments import results

        payment = _payment(checkout_request_id="CO-42")
        db = self._db_with(payment)
        with (
            patch(f"{PAY_RESULTS}.SessionLocal", new=FakeSessionMaker(db)),
            patch(f"{PAY_RESULTS}.apply_payment_result", new=AsyncMock(return_value={"ok": 1})) as apply_result,
        ):
            payload = await results.apply_payment_result_by_checkout(
                "CO-42", status="success"
            )
        self.assertEqual(payload, {"ok": 1})
        apply_result.assert_awaited_once()
        self.assertEqual(apply_result.await_args.args[0], payment.id)

    async def test_callback_for_unknown_checkout_is_ignored(self):
        from app.agents.payments import results

        with (
            patch(f"{PAY_RESULTS}.SessionLocal", new=FakeSessionMaker(FakeDB())),
            patch(f"{PAY_RESULTS}.apply_payment_result", new=AsyncMock()) as apply_result,
        ):
            self.assertIsNone(
                await results.apply_payment_result_by_checkout("CO-none", status="success")
            )
        apply_result.assert_not_called()

    async def test_success_without_a_conversation_is_a_noop(self):
        from app.agents.payments import results

        payment = _payment(conversation_id=None)
        db = self._db_with(payment)
        with patch(f"{PAY_RESULTS}.SessionLocal", new=FakeSessionMaker(db)):
            await results._on_payment_success(payment.id, "R1")
        self.hub.send.assert_not_called()

    async def test_success_sets_needs_email_when_no_email_is_known(self):
        from app.agents.payments import results

        payment = _payment()
        booking = _booking(passenger_email=None)
        conversation = Conversation(id=payment.conversation_id)
        conversation.facts = None
        db = FakeDB()
        db.register(payment, booking, conversation)
        with patch(f"{PAY_RESULTS}.SessionLocal", new=FakeSessionMaker(db)):
            await results._on_payment_success(payment.id, "R1")
        facts = conversation.facts
        self.assertTrue(facts["needs_email"])
        self.assertEqual(facts["pending_email_booking"]["pnr"], booking.pnr)
        sent_types = [call.args[1]["type"] for call in self.hub.send.await_args_list]
        self.assertIn("bot_message", sent_types)


if __name__ == "__main__":
    unittest.main()
