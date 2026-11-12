"""LLM layer: mock determinism, HTTP backend via injected opener."""
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import (LLMError, LLMNotConfigured, MockLLMClient,  # noqa: E402
                     OpenAICompatibleClient, _mock_extract_proposal)


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


class TestMockExtraction(unittest.TestCase):
    def amount(self, text):
        return json.loads(_mock_extract_proposal(text))["amount"]

    def test_invoice_number_is_not_the_amount(self):
        # "INV-1042" mentioned before the amount used to be read as
        # $1,042.00.
        self.assertEqual(self.amount(
            "Pay invoice INV-1042 for $120.00 to "
            "billing@northwind.example"), "120.00")

    def test_thousands_separator_is_not_a_split(self):
        # "$1,200.00" used to be read as $1.00.
        self.assertEqual(self.amount(
            "Pay $1,200.00 to billing@northwind.example"), "1200.00")
        self.assertEqual(self.amount(
            "Pay 1,200 to billing@northwind.example"), "1200")

    def test_count_is_not_the_amount(self):
        # "2 invoices totalling $120" used to be read as $2.00.
        self.assertEqual(self.amount(
            "Pay for 2 invoices totalling $120.00 to "
            "billing@northwind.example"), "120.00")

    def test_decimal_comma_and_bare_amounts(self):
        self.assertEqual(self.amount(
            "Pay 120,50 to billing@northwind.example"), "120.50")
        self.assertEqual(self.amount(
            "Pay 120 to billing@northwind.example"), "120")
        self.assertEqual(self.amount(
            "Pay 120.00 to billing@northwind.example"), "120.00")


class _FakeResponse:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeResponseRaw(_FakeResponse):
    def __init__(self, raw: bytes):
        self._raw = raw


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

    def test_garbage_json_is_an_llm_error(self):
        # A 200 with a non-JSON body used to escape as a raw
        # JSONDecodeError, crashing the flow instead of engaging the
        # checker's fail-closed path.
        client = OpenAICompatibleClient(
            "https://llm.example/v1", "k", "m",
            opener=lambda req, timeout=None: _FakeResponseRaw(b"not json"))
        with self.assertRaises(LLMError):
            client.chat("checker", [])

    def test_null_content_becomes_empty_string(self):
        # Some providers return content: null; it used to flow through
        # as None and crash the agents' JSON parsing with a TypeError.
        client = OpenAICompatibleClient(
            "https://llm.example/v1", "k", "m",
            opener=lambda req, timeout=None: _FakeResponse(
                {"choices": [{"message": {"content": None}}]}))
        self.assertEqual(client.chat("checker", []).content, "")

    def test_null_usage_counts_as_zero(self):
        client = OpenAICompatibleClient(
            "https://llm.example/v1", "k", "m",
            opener=lambda req, timeout=None: _FakeResponse(
                {"choices": [{"message": {"content": "hi"}}],
                 "usage": {"prompt_tokens": None, "completion_tokens": None,
                           "total_tokens": None}}))
        resp = client.chat("checker", [])
        self.assertEqual(resp.usage["total_tokens"], 0)

    def test_misshapen_response_is_an_llm_error(self):
        client = OpenAICompatibleClient(
            "https://llm.example/v1", "k", "m",
            opener=lambda req, timeout=None: _FakeResponse(
                {"choices": [{"message": None}]}))
        # message: null degrades to empty content, not a crash...
        self.assertEqual(client.chat("checker", []).content, "")
        client = OpenAICompatibleClient(
            "https://llm.example/v1", "k", "m",
            opener=lambda req, timeout=None: _FakeResponseRaw(b"[1, 2]"))
        with self.assertRaises(LLMError):
            client.chat("checker", [])

    def test_missing_key_is_not_configured(self):
        with self.assertRaises(LLMNotConfigured):
            OpenAICompatibleClient("https://llm.example/v1", "", "m")


if __name__ == "__main__":
    unittest.main()
