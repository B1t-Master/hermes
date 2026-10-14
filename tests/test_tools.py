import json
import unittest

from app.agents import tools


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


class CallToolTest(unittest.IsolatedAsyncioTestCase):
    async def test_status(self):
        result = await tools.call_tool("get_flight_status", {"flight_no": "KQ100"})
        self.assertTrue(result["found"])

    async def test_rebook(self):
        result = await tools.call_tool("rebook_flight", {"flight_no": "KQ100"})
        self.assertTrue(result["rebooked"])

    async def test_namespaced_tool_name_is_recovered(self):
        result = await tools.call_tool("repo_browser.rebook_flight", {"flight_no": "KQ100"})
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
                                "arguments": json.dumps({"flight_no": "KQ100"}),
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
