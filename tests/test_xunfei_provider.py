import json
import os
import unittest
from unittest.mock import patch

from auto_harness.providers import Message
from auto_harness.providers.xunfei import XunfeiSparkProvider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class XunfeiProviderTests(unittest.TestCase):
    def test_builds_anthropic_payload_and_extracts_text(self):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["timeout"] = timeout
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["headers"] = dict(req.header_items())
            return FakeResponse({"content": [{"type": "text", "text": "hello"}], "usage": {"input_tokens": 1}})

        env = {
            "XUNFEI_API_BASE": "https://example.invalid/anthropic",
            "XUNFEI_API_KEY": "test-key",
            "XUNFEI_MODEL": "test-model",
        }
        with patch.dict(os.environ, env, clear=False):
            provider = XunfeiSparkProvider(urlopen=fake_urlopen)
            result = provider.complete(
                [
                    Message(role="system", content="system prompt"),
                    Message(role="user", content="user prompt"),
                ]
            )

        self.assertEqual(result.text, "hello")
        self.assertEqual(captured["url"], "https://example.invalid/anthropic/v1/messages")
        self.assertEqual(captured["body"]["model"], "test-model")
        self.assertEqual(captured["body"]["system"], "system prompt")
        self.assertEqual(captured["body"]["messages"][0]["content"], "user prompt")
        self.assertIn("X-api-key", captured["headers"])
        self.assertEqual(captured["headers"]["X-api-key"], "test-key")

    def test_extracts_openai_compatible_response(self):
        provider = XunfeiSparkProvider(urlopen=lambda req, timeout: FakeResponse({}))
        text = provider._extract_text({"choices": [{"message": {"content": "ok"}}]})
        self.assertEqual(text, "ok")


if __name__ == "__main__":
    unittest.main()
