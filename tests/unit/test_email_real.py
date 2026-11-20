import tempfile
import unittest
from unittest import mock

from services import email_real


class SendEmailTest(unittest.TestCase):
    def _send(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            return email_real.send_email(
                "pax@example.com", "Subject", "Body line", outbox_dir=tmp, **kwargs
            )

    def test_without_api_key_uses_mock_outbox(self):
        record = self._send(api_key="", sender="onboarding@resend.dev")
        self.assertEqual(record["provider"], "mock")
        self.assertEqual(record["to"], "pax@example.com")

    def test_default_sender_is_resend_test_address(self):
        record = self._send(api_key="")
        self.assertEqual(record["from"], "onboarding@resend.dev")

    def test_resend_success_builds_payload(self):
        with mock.patch("resend.Emails.send", return_value={"id": "abc"}) as send:
            record = self._send(api_key="re_test", sender="onboarding@resend.dev")
        self.assertEqual(record["provider"], "resend")
        self.assertEqual(record["id"], "abc")
        payload = send.call_args.args[0]
        self.assertEqual(payload["to"], ["pax@example.com"])
        self.assertEqual(payload["from"], "onboarding@resend.dev")
        self.assertIn("Body line", payload["text"])
        self.assertIn("Body line", payload["html"])

    def test_resend_failure_falls_back_to_mock(self):
        with mock.patch("resend.Emails.send", side_effect=RuntimeError("nope")):
            record = self._send(api_key="re_test", sender="onboarding@resend.dev")
        self.assertEqual(record["provider"], "mock-fallback")
        self.assertEqual(record["to"], "pax@example.com")


if __name__ == "__main__":
    unittest.main()
