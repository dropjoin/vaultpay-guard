"""Provider-agnostic LLM layer.

VaultPay Guard's agents (proposer, checker) talk to models only
through the :class:`LLMClient` interface, so the AI provider is a
deployment choice, not a code choice:

- :class:`MockLLMClient` — deterministic, offline. Powers the test
  suite and the offline demo. Its default responder understands the
  two agent prompts (draft a structured proposal / return a checker
  verdict) with simple, explainable rules; tests can inject any
  responder they need, including hostile ones.
- :class:`OpenAICompatibleClient` — any endpoint that speaks the
  OpenAI chat-completions shape (OpenAI, OpenRouter, a local server,
  a gateway). Configured purely by env vars: ``LLM_API_KEY``,
  ``LLM_BASE_URL``, ``LLM_MODEL``.

Design rules (carried over from VaultMind's Otari client):
- No key configured -> :class:`LLMNotConfigured` with an actionable
  message; callers degrade explicitly, never fake a model answer.
- Transport failures surface as :class:`LLMError`, so the checker's
  fail-closed path can engage.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

USER_AGENT = "vaultpay-guard/0.1 (paypal-ai-hackathon)"


class LLMError(Exception):
    """The provider returned an error or was unreachable."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class LLMNotConfigured(LLMError):
    """No API key / model configured for the HTTP backend."""


@dataclass
class LLMResponse:
    content: str
    model: str = "mock"
    usage: dict = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


class LLMClient:
    """Interface every backend implements."""

    name = "base"

    def chat(self, role: str, messages: list[dict]) -> LLMResponse:
        raise NotImplementedError


# ----------------------------------------------------------------------
# Mock backend (deterministic, offline)
# ----------------------------------------------------------------------
# Amount tokens: digits with optional thousands commas and an
# optional decimal part ("120.00", "1,200.00", "120,50"). The number
# is matched as one token so a thousands separator can never split
# "$1,200.00" into "$1.00".
_AMOUNT_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d{2})?|\d+,\d{2}")
_CURRENCY_PREFIX_RE = re.compile(
    r"(?:USD|EUR|GBP|\$|€|£)\s*(" + _AMOUNT_TOKEN_RE.pattern + r")")
_DECIMAL_AMOUNT_RE = re.compile(
    r"\b(\d[\d,]*(?:\.\d{2})|\d+,\d{2})\b")
_BARE_AMOUNT_RE = re.compile(r"\b(\d[\d,]*)\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_CURRENCY_HINTS = (("EUR", ("eur", "€")), ("GBP", ("gbp", "£")),
                   ("USD", ("usd", "$")))


def _normalize_amount_token(token: str) -> str:
    """One matched amount token -> the decimal string the proposer
    prompt asks for.

    "1,200.00" -> "1200.00" (thousands commas dropped),
    "120,50"   -> "120.50"  (decimal comma),
    "1,200"    -> "1200"    (thousands comma, no decimals).
    """
    if "," in token and "." in token:
        return token.replace(",", "")
    if "," in token:
        head, _, tail = token.rpartition(",")
        if len(tail) == 2 and "," not in head:
            return f"{head}.{tail}"
        return token.replace(",", "")
    return token


def _extract_amount(text: str) -> str:
    """Pick the payment amount out of the request text.

    Preference order matters: a bare "first number wins" scan reads
    the invoice number ("INV-1042" -> $1,042) or a count ("2
    invoices" -> $2.00) as the amount. So: an amount carrying a
    currency marker wins, then an amount with decimals, then the
    first bare number. The caller strips the invoice reference before
    calling, so it can never be mistaken for money either.
    """
    for pattern in (_CURRENCY_PREFIX_RE, _DECIMAL_AMOUNT_RE,
                    _BARE_AMOUNT_RE):
        m = pattern.search(text)
        if m:
            return _normalize_amount_token(m.group(1))
    return ""


def _mock_extract_proposal(user_text: str) -> str:
    """Deterministic stand-in for model extraction.

    Understands the demo's request shape: an amount, a payee (email or
    ``to <name>``), an optional currency hint and an optional
    ``INV-...`` reference. Returns the same JSON the proposer prompt
    asks a real model for.
    """
    email = _EMAIL_RE.search(user_text)
    payee = email.group(0) if email else ""
    if not payee:
        m = re.search(r"\bto\s+([A-Za-z][\w .'-]{1,40}?)(?=\s+(?:for|amount|inv|invoice)\b|[,.]|$)",
                      user_text, re.IGNORECASE)
        if m:
            payee = m.group(1).strip()
    ref = re.search(r"\bINV-[\w-]+\b", user_text, re.IGNORECASE)
    # Strip the invoice reference before hunting for the amount: its
    # digits are an identifier, not money ("INV-1042" is not $1,042).
    amount_text = user_text
    if ref:
        amount_text = user_text[:ref.start()] + " " + user_text[ref.end():]
    amount = _extract_amount(amount_text)
    currency = "USD"
    lowered = user_text.lower()
    for code, hints in _CURRENCY_HINTS:
        if any(h in lowered for h in hints):
            currency = code
            break
    return json.dumps({
        "payee": payee,
        "amount": amount,
        "currency": currency,
        "reference": ref.group(0) if ref else "",
        "memo": user_text.strip(),
        "rationale": "Mock proposer extracted the payment fields from "
                     "the request text deterministically.",
    })


def default_mock_responder(role: str, messages: list[dict]) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    if "PROPOSER" in system:
        # The request text rides in the user message after the marker.
        marker = "Payment request (untrusted data):"
        request = user.split(marker, 1)[1].strip() if marker in user else user
        return _mock_extract_proposal(request)
    if "CHECKER" in system:
        return json.dumps({
            "verdict": "PASS",
            "reasons": ["Mock checker: proposal fields are consistent "
                        "with the request text and the policy facts."],
        })
    return "Mock model: no role matched."


class MockLLMClient(LLMClient):
    """Deterministic backend. ``responder`` may be any callable
    ``(role, messages) -> str`` — tests use it to simulate hostile or
    broken models (unparseable output, loosening attempts, ...)."""

    name = "mock"

    def __init__(self, responder=None):
        self.responder = responder or default_mock_responder
        self.calls: list[tuple[str, list[dict]]] = []

    def chat(self, role: str, messages: list[dict]) -> LLMResponse:
        self.calls.append((role, messages))
        return LLMResponse(content=self.responder(role, messages),
                           model="mock-deterministic")


# ----------------------------------------------------------------------
# OpenAI-compatible HTTP backend
# ----------------------------------------------------------------------
def _usage_int(value) -> int:
    """Best-effort usage counter: null/garbage counts as 0, never a crash."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class OpenAICompatibleClient(LLMClient):
    """Chat-completions client for any OpenAI-compatible endpoint.

    ``opener`` is injectable for tests; by default it is
    :func:`urllib.request.urlopen`.
    """

    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 60.0, opener=None):
        if not api_key:
            raise LLMNotConfigured(
                "No LLM API key configured. Set LLM_API_KEY in .env, or "
                "run with the mock backend (the default when no key is "
                "present).")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def chat(self, role: str, messages: list[dict]) -> LLMResponse:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        try:
            with self._opener(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace") if exc.fp else ""
            raise LLMError(f"LLM provider HTTP {exc.code}: {raw[:300]}",
                           status=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.HTTPException, OSError) as exc:
            raise LLMError(f"LLM provider unreachable: {exc}") from exc
        try:
            data = json.loads(raw)
            choices = data.get("choices") or []
            if not choices:
                raise LLMError(
                    f"LLM provider returned no choices: {str(data)[:300]}")
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            usage = data.get("usage") or {}
        except LLMError:
            raise
        except (ValueError, AttributeError, TypeError, KeyError) as exc:
            # A 200 with garbage JSON, a null message, or a misshapen
            # payload is a provider failure like any other: surface
            # it as LLMError so the checker's fail-closed path can
            # engage, instead of crashing the flow with a raw
            # JSONDecodeError / AttributeError / TypeError.
            raise LLMError(f"LLM provider returned an unreadable "
                           f"response: {exc}") from exc
        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            usage={
                "prompt_tokens": _usage_int(usage.get("prompt_tokens")),
                "completion_tokens": _usage_int(
                    usage.get("completion_tokens")),
                "total_tokens": _usage_int(usage.get("total_tokens")),
            },
        )


def build_llm_client() -> LLMClient:
    """Factory: real provider when LLM_API_KEY is set, else the mock."""
    if os.environ.get("LLM_API_KEY"):
        return OpenAICompatibleClient(
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ["LLM_API_KEY"],
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        )
    return MockLLMClient()
