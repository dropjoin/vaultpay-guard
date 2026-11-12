"""PayPal sandbox client — Orders API v2, plus a simulated twin.

Two implementations of the same interface:

- :class:`PayPalClient` talks to the real sandbox
  (``https://api-m.sandbox.paypal.com``): OAuth2 client-credentials
  token, create order, capture order, get order. Credentials come
  ONLY from the ``PAYPAL_CLIENT_ID`` / ``PAYPAL_CLIENT_SECRET``
  environment variables — never hardcoded, never logged, never
  printed (error messages deliberately exclude the Authorization
  header and the secret).

- :class:`SimulatedPayPalClient` mimics the same response shapes
  locally so the whole agent flow is testable and demoable offline.
  It is unmistakably labeled: ``mode == "simulated"`` and every
  response carries ``"simulated": True``. It never touches the
  network and no money — real or sandbox — moves.

:func:`build_paypal_client` picks the real client when credentials
are present and the simulated one otherwise.
"""
from __future__ import annotations

import base64
import http.client
import itertools
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from .money import format_amount

USER_AGENT = "vaultpay-guard/0.1 (paypal-ai-hackathon)"
SANDBOX_BASE = "https://api-m.sandbox.paypal.com"
LIVE_BASE = "https://api-m.paypal.com"


class PayPalError(Exception):
    """PayPal returned an error or was unreachable."""

    def __init__(self, message: str, status: int | None = None,
                 detail: str = ""):
        super().__init__(message)
        self.status = status
        self.detail = detail


class PayPalNotConfigured(PayPalError):
    """No sandbox credentials in the environment."""


def approval_url(order: dict) -> str | None:
    """The payer-approval URL from an order's HATEOAS links, if any.

    Orders v2 returns it as ``rel="approve"`` (older responses) or
    ``rel="payer-action"`` (newer ones). The payer must visit it and
    approve before the order can be captured — that gap is exactly
    why live execution is two-phase (see :mod:`src.flow`).
    """
    for link in order.get("links", []) or []:
        if link.get("rel") in ("approve", "payer-action"):
            return link.get("href")
    return None


@dataclass
class PayPalConfig:
    client_id: str
    client_secret: str
    base_url: str = SANDBOX_BASE

    @classmethod
    def from_env(cls) -> "PayPalConfig":
        client_id = os.environ.get("PAYPAL_CLIENT_ID", "")
        client_secret = os.environ.get("PAYPAL_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            raise PayPalNotConfigured(
                "PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET are not set. "
                "Create a sandbox app at developer.paypal.com and put "
                "its credentials in .env (see .env.example), or run in "
                "simulated mode.")
        env = os.environ.get("PAYPAL_ENV", "sandbox").lower()
        return cls(client_id=client_id, client_secret=client_secret,
                   base_url=LIVE_BASE if env == "live" else SANDBOX_BASE)


# A transport is any callable (method, url, headers, body_bytes) ->
# (status:int, parsed_json:dict). Injectable so unit tests exercise the
# real client logic without network access.
#
# The default timeout is a generous 90 s: the sandbox's observed
# latency is spiky (single token/order calls measured between 9 s and
# >30 s on 2026-11-04), and a 30 s read timeout made live calls fail
# intermittently with no actionable difference for the caller.
def _urllib_transport(method: str, url: str, headers: dict,
                      body: bytes | None, timeout: float = 90.0):
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace") if exc.fp else ""
        try:
            parsed = json.loads(raw) if raw else {}
        except ValueError:
            parsed = {"raw": raw}
        return exc.code, parsed
    except (urllib.error.URLError, TimeoutError, ConnectionError,
            http.client.HTTPException, OSError) as exc:
        raise PayPalError(f"PayPal unreachable at {url}: {exc}") from exc


class PayPalClient:
    """Real Orders v2 client (sandbox by default)."""

    @property
    def mode(self) -> str:
        # The mode follows the endpoint actually configured: a client
        # pointed at the live API must never report itself as
        # "sandbox" — the web badges, the ledger records and the grid
        # mode tags all key off this value.
        return "live" if self.config.base_url == LIVE_BASE else "sandbox"

    def __init__(self, config: PayPalConfig, transport=None):
        self.config = config
        self._transport = transport or _urllib_transport
        self._token: str | None = None
        self._token_expires_at = 0.0

    # -- auth ---------------------------------------------------------
    def get_access_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 30:
            return self._token
        basic = base64.b64encode(
            f"{self.config.client_id}:{self.config.client_secret}"
            .encode()).decode()
        status, data = self._transport(
            "POST", f"{self.config.base_url}/v1/oauth2/token",
            {"Authorization": f"Basic {basic}",
             "Content-Type": "application/x-www-form-urlencoded",
             "User-Agent": USER_AGENT},
            b"grant_type=client_credentials")
        if status != 200 or "access_token" not in data:
            raise PayPalError(
                f"PayPal OAuth failed (HTTP {status}): "
                f"{data.get('error_description') or data.get('error') or data}",
                status=status, detail=str(data)[:300])
        self._token = data["access_token"]
        self._token_expires_at = time.time() + int(data.get("expires_in", 3600))
        return self._token

    def _call(self, method: str, path: str, payload: dict | None = None):
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Authorization": f"Bearer {self.get_access_token()}",
                   "Content-Type": "application/json",
                   "User-Agent": USER_AGENT}
        status, data = self._transport(
            method, f"{self.config.base_url}{path}", headers, body)
        if status >= 400:
            name = data.get("name") or data.get("error") or "error"
            msg = data.get("message") or data.get("error_description") or ""
            raise PayPalError(f"PayPal {method} {path} failed "
                              f"(HTTP {status}) {name}: {msg}",
                              status=status, detail=str(data)[:300])
        return data

    # -- Orders v2 ------------------------------------------------------
    def create_order(self, *, amount_minor: int, currency: str,
                     payee_email: str = "", reference: str = "",
                     description: str = "") -> dict:
        unit: dict = {
            "amount": {"currency_code": currency.upper(),
                       "value": format_amount(amount_minor, currency)},
        }
        if reference:
            unit["reference_id"] = reference
        if description:
            unit["description"] = description[:127]
        if payee_email:
            unit["payee"] = {"email_address": payee_email}
        return self._call("POST", "/v2/checkout/orders", {
            "intent": "CAPTURE",
            "purchase_units": [unit],
        })

    def capture_order(self, order_id: str) -> dict:
        return self._call(
            "POST", f"/v2/checkout/orders/{urllib.parse.quote(order_id)}/capture",
            {})

    def get_order(self, order_id: str) -> dict:
        return self._call(
            "GET", f"/v2/checkout/orders/{urllib.parse.quote(order_id)}")


# ----------------------------------------------------------------------
# Simulated client — same interface, same shapes, zero network.
# ----------------------------------------------------------------------
@dataclass
class SimulatedPayPalClient:
    mode = "simulated"
    _orders: dict = field(default_factory=dict)
    _counter: itertools.count = field(default_factory=lambda: itertools.count(1))

    def _new_id(self, prefix: str) -> str:
        return f"SIM-{prefix}-{next(self._counter):04d}"

    def create_order(self, *, amount_minor: int, currency: str,
                     payee_email: str = "", reference: str = "",
                     description: str = "") -> dict:
        order_id = self._new_id("ORDER")
        order = {
            "id": order_id,
            "status": "CREATED",
            "intent": "CAPTURE",
            "purchase_units": [{
                "reference_id": reference or "default",
                "amount": {"currency_code": currency.upper(),
                           "value": format_amount(amount_minor, currency)},
                **({"payee": {"email_address": payee_email}}
                   if payee_email else {}),
                **({"description": description} if description else {}),
            }],
            "links": [{
                "href": f"https://www.sandbox.paypal.com/checkoutnow?token={order_id}",
                "rel": "approve", "method": "GET"}],
            "simulated": True,
        }
        self._orders[order_id] = order
        return order

    def capture_order(self, order_id: str) -> dict:
        order = self._orders.get(order_id)
        if order is None:
            raise PayPalError(f"Simulated order not found: {order_id}",
                              status=404, detail="RESOURCE_NOT_FOUND")
        if order["status"] == "COMPLETED":
            return order
        unit = order["purchase_units"][0]
        capture = {
            "id": self._new_id("CAPTURE"),
            "status": "COMPLETED",
            "amount": unit["amount"],
            "final_capture": True,
        }
        captured = {
            **order,
            "status": "COMPLETED",
            "purchase_units": [{
                **unit,
                "payments": {"captures": [capture]},
            }],
        }
        self._orders[order_id] = captured
        return captured

    def get_order(self, order_id: str) -> dict:
        order = self._orders.get(order_id)
        if order is None:
            raise PayPalError(f"Simulated order not found: {order_id}",
                              status=404, detail="RESOURCE_NOT_FOUND")
        return order


def build_paypal_client() -> PayPalClient | SimulatedPayPalClient:
    """Real sandbox client when credentials exist, simulated otherwise."""
    try:
        return PayPalClient(PayPalConfig.from_env())
    except PayPalNotConfigured:
        return SimulatedPayPalClient()
