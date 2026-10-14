import json
import unittest
from unittest import mock

from app.agents import tools
from app.config import settings

TICKET = "1234567890"


class DetectRebookTest(unittest.TestCase):
    def test_matches_rebook_phrases(self):
        for query in (
            "please rebook me",
            "can I change my flight?",
            "move me to the next flight",
            "I need an alternative flight",
        ):
            self.assertTrue(tools.detect_rebook_request(query), query)

    def test_ignores_unrelated(self):
        for query in ("what is the baggage allowance", "is KQ100 delayed", "hello"):
            self.assertFalse(tools.detect_rebook_request(query), query)

    def test_ignores_informational_rebooking_questions(self):
        for query in (
            "Is the KQ Self Rebooking service free?",
            "What do I need to rebook myself?",
            "Can I rebook a flight that has not been disrupted?",
            "How far in advance can I rebook with the self-rebooking tool?",
            "What happens if I do not rebook myself?",
            "Does the self-rebooking tool rebook award tickets?",
        ):
            self.assertFalse(tools.detect_rebook_request(query), query)

    def test_follow_up_ticket_after_rebook_keeps_tools(self):
        history = [{"role": "user", "content": "please rebook my cancelled flight"}]
        self.assertTrue(
            tools.detect_rebook_request("My booking reference is 1234567890.", history)
        )

    def test_ticket_follow_up_without_rebook_history(self):
        history = [{"role": "user", "content": "hello there"}]
        self.assertFalse(
            tools.detect_rebook_request("My booking reference is 1234567890.", history)
        )

    def test_unrelated_follow_up_after_rebook(self):
        history = [{"role": "user", "content": "please rebook my cancelled flight"}]
        self.assertFalse(tools.detect_rebook_request("What is the baggage allowance?", history))


class CallToolTest(unittest.IsolatedAsyncioTestCase):
    async def test_status(self):
        result = await tools.call_tool("get_flight_status", {"flight_no": "KQ100"})
        self.assertTrue(result["found"])

    async def test_rebook_requires_ticket(self):
        result = await tools.call_tool("rebook_flight", {"flight_no": "KQ100"})
        self.assertFalse(result["rebooked"])
        self.assertTrue(result["needs_ticket"])

    async def test_rebook_emails_mapped_address(self):
        fake = {"sent_at": "2025-03-01T00:00:00+00:00", "provider": "resend", "id": "abc"}
        with (
            mock.patch.object(tools, "_lookup_ticket_email", return_value="pax@example.com"),
            mock.patch.object(tools.email_real, "send_email", return_value=fake) as send,
        ):
            result = await tools.call_tool(
                "rebook_flight", {"flight_no": "KQ100", "ticket_number": TICKET}
            )
        self.assertTrue(result["rebooked"])
        self.assertTrue(result["email_sent"])
        self.assertEqual(result["email_to"], "pax@example.com")
        self.assertEqual(result["email_provider"], "resend")
        send.assert_called_once()
        self.assertEqual(send.call_args.args[0], "pax@example.com")

    async def test_rebook_falls_back_to_default_email(self):
        fake = {"sent_at": "t", "provider": "resend"}
        with (
            mock.patch.object(tools, "_lookup_ticket_email", return_value=None),
            mock.patch.object(tools.email_real, "send_email", return_value=fake),
        ):
            result = await tools.call_tool(
                "rebook_flight", {"flight_no": "KQ100", "ticket_number": TICKET}
            )
        self.assertTrue(result["email_sent"])
        self.assertEqual(result["email_to"], settings.rebook_confirmation_email)

    async def test_failed_email_does_not_break_rebook(self):
        with (
            mock.patch.object(tools, "_lookup_ticket_email", return_value=None),
            mock.patch.object(tools.email_real, "send_email", side_effect=RuntimeError("boom")),
        ):
            result = await tools.call_tool(
                "rebook_flight", {"flight_no": "KQ100", "ticket_number": TICKET}
            )
        self.assertTrue(result["rebooked"])
        self.assertNotIn("email_sent", result)

    async def test_namespaced_tool_name_is_recovered(self):
        with (
            mock.patch.object(tools, "_lookup_ticket_email", return_value=None),
            mock.patch.object(
                tools.email_real, "send_email", return_value={"sent_at": "t", "provider": "resend"}
            ),
        ):
            result = await tools.call_tool(
                "repo_browser.rebook_flight",
                {"flight_no": "KQ100", "ticket_number": TICKET},
            )
        self.assertTrue(result["rebooked"])

    async def test_unknown_tool(self):
        result = await tools.call_tool("nope", {})
        self.assertIn("unknown tool", result["error"])


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.seen = []

    async def tool_message(self, messages, tools, **kwargs):
        self.seen.append(list(messages))
        return self._responses.pop(0)

    async def stream(self, messages, **kwargs):
        yield ""

    async def complete(self, messages, **kwargs):
        return ""


class RunToolTurnTest(unittest.IsolatedAsyncioTestCase):
    async def test_executes_tool_then_returns_answer(self):
        fake = _FakeLLM(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "rebook_flight",
                                "arguments": json.dumps(
                                    {"flight_no": "KQ100", "ticket_number": TICKET}
                                ),
                            },
                        }
                    ],
                },
                {"role": "assistant", "content": "Done: reference HRM-ABC123.", "tool_calls": []},
            ]
        )
        tokens: list[str] = []

        async def on_token(token):
            tokens.append(token)

        with (
            mock.patch.object(tools, "_lookup_ticket_email", return_value=None),
            mock.patch.object(
                tools.email_real, "send_email", return_value={"sent_at": "t", "provider": "resend"}
            ),
        ):
            answer = await tools.run_tool_turn(
                [{"role": "user", "content": "rebook KQ100"}], llm=fake, on_token=on_token
            )
        self.assertIn("HRM-", answer)
        self.assertTrue(tokens)
        tool_messages = [m for m in fake.seen[1] if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("reference", tool_messages[0]["content"])

    async def test_direct_answer_without_tool(self):
        fake = _FakeLLM([{"role": "assistant", "content": "Sure, I can help.", "tool_calls": []}])
        answer = await tools.run_tool_turn([{"role": "user", "content": "hi"}], llm=fake)
        self.assertEqual(answer, "Sure, I can help.")
        self.assertEqual(len(fake.seen), 1)


if __name__ == "__main__":
    unittest.main()
