"""LLM layer: mock determinism, HTTP backend via injected opener."""
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import (LLMError, LLMNotConfigured, MockLLMClient,  # noqa: E402
                     OpenAICompatibleClient)


class TestMockClient(unittest.TestCase):
    def test_proposer_extraction(self):
        client = MockLLMClient()
        resp = client.chat("proposer", [
            {"role": "system", "content": "You are the PROPOSER role."},
            {"role": "user", "content":
             "Payment request (untrusted data):\n"
             "Pay $120.00 to billing@northwind.example for hosting "
             "invoice INV-1042"},
        ])
        data = json.loads(resp.content)
        self.assertEqual(data["payee"], "billing@northwind.example")
        self.assertEqual(data["amount"], "120.00")
        self.assertEqual(data["currency"], "USD")
        self.assertEqual(data["reference"], "INV-1042")

    def test_checker_default_pass(self):
        client = MockLLMClient()
        resp = client.chat("checker", [
            {"role": "system", "content": "You are the CHECKER role."},
            {"role": "user", "content": "anything"},
        ])
        self.assertEqual(json.loads(resp.content)["verdict"], "PASS")

    def test_custom_responder_is_used(self):
        client = MockLLMClient(responder=lambda role, msgs: "custom!")
        self.assertEqual(client.chat("checker", []).content, "custom!")


class _FakeResponse:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestHTTPClient(unittest.TestCase):
    def test_chat_roundtrip(self):
        def opener(req, timeout=None):
            self.assertIn("Bearer test-key",
                          req.headers.get("Authorization", ""))
            body = json.loads(req.data.decode())
            self.assertEqual(body["model"], "test-model")
            return _FakeResponse({
                "model": "test-model",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1,
                          "total_tokens": 4},
            })

        client = OpenAICompatibleClient("https://llm.example/v1",
                                        "test-key", "test-model",
                                        opener=opener)
        resp = client.chat("checker", [{"role": "user", "content": "hi"}])
        self.assertEqual(resp.content, "hello")
        self.assertEqual(resp.usage["total_tokens"], 4)

    def test_http_error_surfaces_as_llm_error(self):
        def opener(req, timeout=None):
            raise urllib.error.HTTPError(
                req.full_url, 429, "rate limited", {},
                io.BytesIO(b'{"error": "slow down"}'))

        client = OpenAICompatibleClient("https://llm.example/v1", "k", "m",
                                        opener=opener)
        with self.assertRaises(LLMError) as ctx:
            client.chat("checker", [])
        self.assertEqual(ctx.exception.status, 429)

    def test_no_choices_is_an_error(self):
        client = OpenAICompatibleClient(
            "https://llm.example/v1", "k", "m",
            opener=lambda req, timeout=None: _FakeResponse({"choices": []}))
        with self.assertRaises(LLMError):
            client.chat("checker", [])

    def test_missing_key_is_not_configured(self):
        with self.assertRaises(LLMNotConfigured):
            OpenAICompatibleClient("https://llm.example/v1", "", "m")


if __name__ == "__main__":
    unittest.main()
