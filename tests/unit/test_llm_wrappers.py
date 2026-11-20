"""LLM fallback chain, intent classification and sentiment analyser tests."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents import intent as intent_module
from app.agents import llm as llm_module
from app.agents import sentiment as sentiment_module
from app.agents.llm import LLM, LLMError


def _response(content="The answer.", tool_calls=None):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _fake_client(create=AsyncMock(return_value=_response())):
    client = MagicMock()
    client.chat.completions.create = create
    return client


class _SettingsKeys:
    """Context helper: set LLM keys on the settings object."""

    def __init__(self, groq="", deepseek="", reasoning=""):
        self.values = {
            "groq_api_key": groq,
            "deepseek_api_key": deepseek,
            "llm_reasoning_effort": reasoning,
        }

    def __enter__(self):
        self.patchers = [
            patch(f"app.config.settings.{key}", value)
            for key, value in self.values.items()
        ]
        for patcher in self.patchers:
            patcher.start()
        return self

    def __exit__(self, *exc_info):
        for patcher in self.patchers:
            patcher.stop()


class LLMChainTest(unittest.IsolatedAsyncioTestCase):
    async def test_no_providers_raises(self):
        llm = LLM()
        with _SettingsKeys():
            with self.assertRaises(LLMError):
                await llm.complete([{"role": "user", "content": "hi"}])
            with self.assertRaises(LLMError):
                await llm.tool_message([{"role": "user", "content": "hi"}], tools=[])
            with self.assertRaises(LLMError):
                async for _ in llm.stream([{"role": "user", "content": "hi"}]):
                    pass

    async def test_chain_order_groq_primary_fallback_then_deepseek(self):
        with _SettingsKeys(groq="gsk", deepseek="ds", reasoning="low"):
            llm = LLM()
            chain = llm._chain()
        self.assertEqual(len(chain), 3)
        clients = [client for client, _ in chain]
        self.assertIs(clients[0], clients[1])  # same Groq client, two models
        self.assertIsNot(clients[0], clients[2])

    async def test_chain_is_cached(self):
        with _SettingsKeys(groq="gsk"):
            llm = LLM()
            first = llm._chain()
            second = llm._chain()
        self.assertIs(first, second)

    async def test_provider_kwargs_only_for_openai_models(self):
        with _SettingsKeys(groq="gsk", reasoning="low"):
            llm = LLM()
            self.assertEqual(
                llm._provider_kwargs("openai/gpt-oss-120b"), {"reasoning_effort": "low"}
            )
            self.assertEqual(llm._provider_kwargs("deepseek-chat"), {})

    async def test_complete_walks_the_chain_on_errors(self):
        with _SettingsKeys(groq="gsk", deepseek="ds"):
            llm = LLM()
            flaky = _fake_client(AsyncMock(side_effect=RuntimeError("503")))
            good = _fake_client()
            llm._chain_cache = [(flaky, "openai/primary"), (good, "deepseek-chat")]
            text = await llm.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(text, "The answer.")
        self.assertEqual(good.chat.completions.create.await_count, 1)

    async def test_complete_raises_after_the_chain_is_exhausted(self):
        with _SettingsKeys(groq="gsk"):
            llm = LLM()
            client = _fake_client(AsyncMock(side_effect=RuntimeError("503")))
            llm._chain_cache = [(client, "openai/primary")]
            with self.assertRaises(LLMError):
                await llm.complete([{"role": "user", "content": "hi"}])

    async def test_empty_response_counts_as_a_failure(self):
        with _SettingsKeys(groq="gsk"):
            llm = LLM()
            empty = _fake_client(AsyncMock(return_value=_response(content="")))
            llm._chain_cache = [(empty, "openai/primary")]
            with self.assertRaises(LLMError):
                await llm.complete([{"role": "user", "content": "hi"}])

    async def test_tool_message_shapes_tool_calls(self):
        tool_call = MagicMock()
        tool_call.id = "call-1"
        tool_call.function.name = "create_booking"
        tool_call.function.arguments = "{}"
        with _SettingsKeys(groq="gsk"):
            llm = LLM()
            client = _fake_client(AsyncMock(return_value=_response(tool_calls=[tool_call])))
            llm._chain_cache = [(client, "openai/primary")]
            message = await llm.tool_message(
                [{"role": "user", "content": "book"}], tools=[{"x": 1}]
            )
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "create_booking")
        create = client.chat.completions.create
        self.assertEqual(create.await_args.kwargs["tool_choice"], "auto")

    async def test_stream_yields_deltas(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Hello "

        async def stream_gen():
            yield chunk
            yield chunk

        client = _fake_client(AsyncMock(return_value=stream_gen()))
        llm = LLM()
        llm._chain_cache = [(client, "openai/primary")]
        parts = [part async for part in llm.stream([{"role": "user", "content": "hi"}])]
        self.assertEqual(parts, ["Hello ", "Hello "])

    async def test_stream_falls_back_to_the_next_provider(self):
        async def fail_stream():
            raise RuntimeError("dead")
            yield  # pragma: no cover

        dead = _fake_client(AsyncMock(return_value=fail_stream()))

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "ok"

        async def ok_stream():
            yield chunk

        alive = _fake_client(AsyncMock(return_value=ok_stream()))
        llm = LLM()
        llm._chain_cache = [(dead, "openai/primary"), (alive, "deepseek-chat")]
        parts = [part async for part in llm.stream([{"role": "user", "content": "hi"}])]
        self.assertEqual(parts, ["ok"])

    async def test_get_llm_returns_a_singleton(self):
        with patch("app.agents.llm._llm", None):
            from app.agents.llm import get_llm

            self.assertIs(get_llm(), get_llm())


class IntentTest(unittest.IsolatedAsyncioTestCase):
    def test_hard_safety_triggers(self):
        self.assertEqual(
            intent_module.keyword_hard_trigger("There is a BOMB here"), "bomb"
        )
        self.assertIsNone(intent_module.keyword_hard_trigger("all good"))
        self.assertIsNone(intent_module.keyword_hard_trigger(""))

    def test_parse_llm_intent(self):
        self.assertEqual(intent_module.parse_llm_intent("compensation"), "compensation")
        self.assertEqual(intent_module.parse_llm_intent("Intent: safety."), "safety")
        self.assertIsNone(intent_module.parse_llm_intent("banana"))
        self.assertIsNone(intent_module.parse_llm_intent(""))

    async def test_safety_keyword_short_circuits_the_llm(self):
        with patch("app.agents.intent.get_llm") as get_llm:
            result = await intent_module.classify_intent("we have an emergency")
        self.assertEqual(result.intent, "safety")
        self.assertEqual(result.source, "keyword")
        self.assertEqual(result.matched_term, "emergency")
        get_llm.assert_not_called()

    async def test_unconfigured_llm_defaults_to_routine(self):
        with (
            patch("app.agents.intent.get_llm", return_value=MagicMock(is_configured=False)),
        ):
            result = await intent_module.classify_intent("what is the baggage limit?")
        self.assertEqual(result.intent, "routine")
        self.assertEqual(result.source, "default")

    async def test_llm_classification_success_and_failures(self):
        good_llm = MagicMock(is_configured=True, complete=AsyncMock(return_value="human_request"))
        with patch("app.agents.intent.get_llm", return_value=good_llm):
            result = await intent_module.classify_intent("let me talk to a person")
        self.assertEqual(result.intent, "human_request")
        self.assertEqual(result.source, "llm")

        bad_llm = MagicMock(is_configured=True, complete=AsyncMock(return_value="gibberish!"))
        with patch("app.agents.intent.get_llm", return_value=bad_llm):
            result = await intent_module.classify_intent("hello")
        self.assertEqual(result.intent, "routine")
        self.assertEqual(result.source, "default")

        raising_llm = MagicMock(is_configured=True, complete=AsyncMock(side_effect=RuntimeError))
        with patch("app.agents.intent.get_llm", return_value=raising_llm):
            result = await intent_module.classify_intent("hello")
        self.assertEqual(result.intent, "routine")
        self.assertEqual(result.source, "default")


class _FakePipeline:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return [self.predictions]


class SentimentTest(unittest.TestCase):
    def test_empty_text_is_neutral(self):
        analyser = sentiment_module.SentimentAnalyser()
        result = analyser.analyse("")
        self.assertEqual((result.label, result.score), ("neutral", 0.0))

    def test_signed_score_from_label_probabilities(self):
        analyser = sentiment_module.SentimentAnalyser()
        pipeline = _FakePipeline([
            {"label": "negative", "score": 0.9},
            {"label": "neutral", "score": 0.08},
            {"label": "positive", "score": 0.02},
        ])
        with (
            patch.object(sentiment_module.SentimentAnalyser, "available", True),
            patch.object(analyser, "_load", return_value=pipeline),
        ):
            result = analyser.analyse("this is terrible\n\n and awful  ")
        self.assertEqual(result.label, "negative")
        self.assertEqual(result.score, -0.88)
        self.assertEqual(result.confidence, 0.9)
        # Text is whitespace-collapsed and truncated to 1024 chars.
        self.assertEqual(pipeline.calls[0], "this is terrible and awful")

    async def test_analyse_async_runs_off_the_loop(self):
        analyser = sentiment_module.SentimentAnalyser()
        pipeline = _FakePipeline([
            {"label": "positive", "score": 0.9},
            {"label": "neutral", "score": 0.1},
        ])
        with (
            patch.object(sentiment_module.SentimentAnalyser, "available", True),
            patch.object(analyser, "_load", return_value=pipeline),
        ):
            result = await analyser.analyse_async("great service")
        self.assertEqual(result.label, "positive")
        self.assertEqual(result.score, 0.9)

    def test_pipeline_failure_degrades_to_neutral(self):
        analyser = sentiment_module.SentimentAnalyser()

        def boom(text):
            raise RuntimeError("model exploded")

        with (
            patch.object(sentiment_module.SentimentAnalyser, "available", True),
            patch.object(analyser, "_load", side_effect=boom),
        ):
            result = analyser.analyse("angry message")
        self.assertEqual((result.label, result.score), ("neutral", 0.0))

    def test_singleton_caching(self):
        with patch("app.agents.sentiment._analyser", None):
            analyser = sentiment_module.get_sentiment_analyser()
            self.assertIs(analyser, sentiment_module.get_sentiment_analyser())


if __name__ == "__main__":
    unittest.main()
