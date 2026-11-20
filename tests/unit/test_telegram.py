import unittest

import httpx

from app.channels.telegram import (
    HELP,
    WELCOME,
    TelegramAPI,
    TelegramChannel,
    TelegramError,
    _HubSink,
    chunk_text,
    parse_incoming,
    to_telegram_html,
)


class ParseIncomingTest(unittest.TestCase):
    def test_extracts_text_message(self):
        update = {
            "update_id": 1,
            "message": {
                "message_id": 5,
                "from": {"id": 42, "first_name": "Amina"},
                "chat": {"id": 42, "first_name": "Amina"},
                "text": "  is rebooking free?  ",
            },
        }
        incoming = parse_incoming(update)
        self.assertIsNotNone(incoming)
        self.assertEqual(incoming.chat_id, 42)
        self.assertEqual(incoming.text, "is rebooking free?")
        self.assertEqual(incoming.name, "Amina")

    def test_group_chat_falls_back_to_title(self):
        update = {"message": {"chat": {"id": -100, "title": "KQ Fans"}, "text": "hi"}}
        incoming = parse_incoming(update)
        self.assertEqual(incoming.name, "KQ Fans")

    def test_missing_chat_returns_none(self):
        self.assertIsNone(parse_incoming({"message": {"text": "hi"}}))

    def test_non_text_message_returns_none(self):
        update = {"message": {"chat": {"id": 1}, "photo": [{"file_id": "x"}]}}
        self.assertIsNone(parse_incoming(update))

    def test_blank_text_returns_none(self):
        self.assertIsNone(parse_incoming({"message": {"chat": {"id": 1}, "text": "   "}}))

    def test_edited_message_is_ignored(self):
        update = {"edited_message": {"chat": {"id": 1}, "text": "hi"}}
        self.assertIsNone(parse_incoming(update))


class ChunkTextTest(unittest.TestCase):
    def test_short_text_is_single_chunk(self):
        self.assertEqual(chunk_text("hello"), ["hello"])

    def test_long_text_is_split_and_lossless(self):
        text = "word " * 2000  # ~10000 chars
        chunks = chunk_text(text)
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(c) <= 3900 for c in chunks))
        self.assertEqual("".join(chunks), text)

    def test_prefers_newline_boundary(self):
        text = "a" * 3000 + "\n" + "b" * 3000
        chunks = chunk_text(text)
        self.assertEqual(chunks[0], "a" * 3000)
        self.assertEqual(chunks[1], "b" * 3000)


class ToTelegramHtmlTest(unittest.TestCase):
    def test_escapes_html_metacharacters(self):
        self.assertEqual(to_telegram_html("a < b & c > d"), "a &lt; b &amp; c &gt; d")

    def test_converts_bold(self):
        self.assertEqual(to_telegram_html("say **hi** now"), "say <b>hi</b> now")

    def test_plain_text_unchanged(self):
        self.assertEqual(to_telegram_html("no markup"), "no markup")


class _Handler:
    """Records requests and replays canned Bot API responses."""

    def __init__(self, ok=True, result=None, description=None):
        self.requests: list[httpx.Request] = []
        self._ok = ok
        self._result = result
        self._description = description

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = {"ok": self._ok}
        if self._ok:
            body["result"] = self._result
        else:
            body["description"] = self._description
        return httpx.Response(200, json=body)


def _api(handler: _Handler) -> TelegramAPI:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TelegramAPI("TESTTOKEN", base_url="https://api.telegram.test", client=client)


class TelegramAPITest(unittest.IsolatedAsyncioTestCase):
    async def test_get_updates_posts_offset_and_allowed_updates(self):
        handler = _Handler(result=[{"update_id": 7}])
        api = _api(handler)
        updates = await api.get_updates(offset=8, timeout=25)
        self.assertEqual(updates, [{"update_id": 7}])
        request = handler.requests[0]
        self.assertTrue(request.url.path.endswith("/botTESTTOKEN/getUpdates"))
        import json

        payload = json.loads(request.content)
        self.assertEqual(payload["offset"], 8)
        self.assertEqual(payload["timeout"], 25)
        self.assertEqual(payload["allowed_updates"], ["message"])

    async def test_send_message_includes_parse_mode(self):
        handler = _Handler(result={"message_id": 1})
        api = _api(handler)
        await api.send_message(123, "<b>hi</b>", parse_mode="HTML")
        import json

        payload = json.loads(handler.requests[0].content)
        self.assertEqual(payload["chat_id"], 123)
        self.assertEqual(payload["parse_mode"], "HTML")

    async def test_ok_false_raises(self):
        handler = _Handler(ok=False, description="Bad Request: chat not found")
        api = _api(handler)
        with self.assertRaises(TelegramError):
            await api.send_message(1, "x")


class _FakeAPI:
    def __init__(self):
        self.sent: list[tuple[int, str]] = []
        self.actions: list[tuple[int, str]] = []

    async def send_message(self, chat_id, text, parse_mode=None):
        self.sent.append((chat_id, text))
        return {}

    async def send_chat_action(self, chat_id, action="typing"):
        self.actions.append((chat_id, action))

    async def get_updates(self, offset, timeout):
        return []

    async def get_me(self):
        return {"username": "test_bot"}

    async def delete_webhook(self):
        return {}

    async def close(self):
        pass


class TelegramChannelCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_command_sends_welcome(self):
        api = _FakeAPI()
        channel = TelegramChannel(api)
        update = {"message": {"chat": {"id": 9}, "text": "/start"}}
        await channel._handle_update(update)
        self.assertEqual(api.sent, [(9, WELCOME)])

    async def test_help_command_with_bot_suffix(self):
        api = _FakeAPI()
        channel = TelegramChannel(api)
        update = {"message": {"chat": {"id": 9}, "text": "/help@hermes_bot"}}
        await channel._handle_update(update)
        self.assertEqual(api.sent, [(9, HELP)])

    async def test_non_text_update_is_ignored(self):
        api = _FakeAPI()
        channel = TelegramChannel(api)
        await channel._handle_update({"message": {"chat": {"id": 9}, "photo": []}})
        self.assertEqual(api.sent, [])
        self.assertEqual(api.actions, [])


class HubSinkTest(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_only_agent_messages(self):
        seen: list[str] = []

        async def on_agent_message(content: str) -> None:
            seen.append(content)

        sink = _HubSink(on_agent_message)
        await sink.send_json({"type": "bot_message", "message": {"content": "bot"}})
        await sink.send_json({"type": "passenger_message", "message": {"content": "pax"}})
        await sink.send_json({"type": "agent_message", "message": {"content": "human"}})
        await sink.send_json({"type": "agent_message", "message": {"content": ""}})
        self.assertEqual(seen, ["human"])


if __name__ == "__main__":
    unittest.main()
