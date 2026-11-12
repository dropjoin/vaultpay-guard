"""PayPal client: real client via fake transport, simulated twin."""
import base64
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paypal_client import (LIVE_BASE, PayPalClient, PayPalConfig,  # noqa: E402
                               PayPalError, SimulatedPayPalClient)


class FakeTransport:
    """Records calls and answers like the sandbox would."""

    def __init__(self):
        self.calls = []
        self.token_calls = 0

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        if url.endswith("/v1/oauth2/token"):
            self.token_calls += 1
            auth = headers["Authorization"]
            assert auth.startswith("Basic ")
            decoded = base64.b64decode(auth[6:]).decode()
            assert decoded == "cid:csecret"
            return 200, {"access_token": "tok-123", "expires_in": 3600}
        if url.endswith("/v2/checkout/orders"):
            payload = json.loads(body.decode())
            unit = payload["purchase_units"][0]
            return 201, {"id": "ORDER-1", "status": "CREATED",
                         "purchase_units": [unit]}
        if url.endswith("/capture"):
            return 200, {"id": "ORDER-1", "status": "COMPLETED",
                         "purchase_units": [{"payments": {"captures": [
                             {"id": "CAP-1", "status": "COMPLETED"}]}}]}
        if "/v2/checkout/orders/" in url:
            return 200, {"id": "ORDER-1", "status": "COMPLETED"}
        raise AssertionError(f"unexpected URL {url}")


def make_client(transport) -> PayPalClient:
    return PayPalClient(PayPalConfig(client_id="cid",
                                     client_secret="csecret"),
                        transport=transport)


class TestRealClient(unittest.TestCase):
    def test_mode_follows_the_configured_endpoint(self):
        # A client pointed at the live API used to report mode
        # "sandbox", so the web badges and ledger records mislabeled
        # live money as sandbox money.
        live = PayPalClient(PayPalConfig(client_id="cid",
                                         client_secret="csecret",
                                         base_url=LIVE_BASE))
        self.assertEqual(live.mode, "live")
        sandbox = make_client(FakeTransport())
        self.assertEqual(sandbox.mode, "sandbox")

    def test_oauth_token_cached(self):
        t = FakeTransport()
        client = make_client(t)
        self.assertEqual(client.get_access_token(), "tok-123")
        self.assertEqual(client.get_access_token(), "tok-123")
        self.assertEqual(t.token_calls, 1)

    def test_create_capture_get_flow(self):
        client = make_client(FakeTransport())
        order = client.create_order(amount_minor=120_00, currency="USD",
                                    payee_email="billing@northwind.example",
                                    reference="INV-1042")
        self.assertEqual(order["status"], "CREATED")
        self.assertEqual(order["purchase_units"][0]["amount"]["value"],
                         "120.00")
        captured = client.capture_order(order["id"])
        self.assertEqual(captured["status"], "COMPLETED")
        fetched = client.get_order(order["id"])
        self.assertEqual(fetched["id"], "ORDER-1")

    def test_api_error_raises_without_leaking_secret(self):
        def bad_transport(method, url, headers, body):
            if url.endswith("/v1/oauth2/token"):
                return 200, {"access_token": "tok", "expires_in": 3600}
            return 422, {"name": "UNPROCESSABLE_ENTITY",
                         "message": "bad request"}

        client = make_client(bad_transport)
        with self.assertRaises(PayPalError) as ctx:
            client.create_order(amount_minor=100, currency="USD")
        self.assertEqual(ctx.exception.status, 422)
        self.assertNotIn("csecret", str(ctx.exception))
        self.assertNotIn("csecret", ctx.exception.detail)


class TestSimulatedClient(unittest.TestCase):
    def test_full_lifecycle_is_labeled_simulated(self):
        client = SimulatedPayPalClient()
        self.assertEqual(client.mode, "simulated")
        order = client.create_order(amount_minor=50_00, currency="USD",
                                    payee_email="a@b.example")
        self.assertTrue(order["simulated"])
        self.assertEqual(order["status"], "CREATED")
        captured = client.capture_order(order["id"])
        self.assertEqual(captured["status"], "COMPLETED")
        self.assertEqual(
            captured["purchase_units"][0]["payments"]["captures"][0]
            ["amount"]["value"], "50.00")
        self.assertEqual(client.get_order(order["id"])["status"],
                         "COMPLETED")

    def test_unknown_order_raises(self):
        client = SimulatedPayPalClient()
        with self.assertRaises(PayPalError):
            client.capture_order("SIM-ORDER-9999")
        with self.assertRaises(PayPalError):
            client.get_order("SIM-ORDER-9999")


if __name__ == "__main__":
    unittest.main()
