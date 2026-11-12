"""Two-phase live execution — real client over a fake transport.

Against the real Orders v2 API, create_order returns a CREATED order
and capture before payer approval fails. These tests pin the honest
behavior:

  - a policy PASS ends phase 1 as execution ORDER_CREATED (verdict
    stays PASS — never an execution REJECT), with the payer approval
    URL recorded;
  - an ORDER_CREATED order already counts against the daily cap;
  - capture is a separate step that appends a kind="capture" record,
    completes the order, and does NOT double-count the daily cap;
  - capture is idempotent, a failed capture is recorded and
    retryable, and unknown / not-awaiting orders raise CaptureError;
  - the web layer reconciles the pair into one ledger row and exposes
    POST /api/capture.
"""
import base64
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.flow import CaptureError, PaymentFlow, render_card  # noqa: E402
from src.ledger import AuditLedger  # noqa: E402
from src.llm import MockLLMClient  # noqa: E402
from src.paypal_client import (PayPalClient, PayPalConfig,  # noqa: E402
                               SimulatedPayPalClient)
from src.policy import (AccountState, Payee, PolicyConfig)  # noqa: E402
from src.web import VaultPayWebApp, make_handler  # noqa: E402

HOSTING = Payee(id="vendor-hosting", name="Northwind Hosting",
                email="billing@northwind.example")
APPROVE_URL = ("https://www.sandbox.paypal.com/checkoutnow"
               "?token=ORDER-1")


class FakeLiveTransport:
    """Answers like the real sandbox: create -> CREATED + approve
    link; capture only succeeds once `approved` is flipped on (the
    payer step itself happens on PayPal's site, outside this API)."""

    def __init__(self):
        self.calls = []
        self.capture_calls = 0
        self.approved = False

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url))
        if url.endswith("/v1/oauth2/token"):
            auth = headers["Authorization"]
            assert base64.b64decode(auth[6:]).decode() == "cid:csecret"
            return 200, {"access_token": "tok-123", "expires_in": 3600}
        if url.endswith("/v2/checkout/orders"):
            unit = json.loads(body.decode())["purchase_units"][0]
            return 201, {
                "id": "ORDER-1", "status": "CREATED",
                "purchase_units": [unit],
                "links": [{"href": APPROVE_URL, "rel": "approve",
                           "method": "GET"}]}
        if url.endswith("/capture"):
            self.capture_calls += 1
            if not self.approved:
                return 422, {"name": "UNPROCESSABLE_ENTITY",
                             "message": "ORDER_NOT_APPROVED"}
            return 200, {"id": "ORDER-1", "status": "COMPLETED",
                         "purchase_units": [{"payments": {"captures": [
                             {"id": "CAP-1", "status": "COMPLETED"}]}}]}
        raise AssertionError(f"unexpected URL {url}")


def make_live_flow(transport) -> PaymentFlow:
    cfg = PolicyConfig(per_tx_cap_minor=250_00, daily_cap_minor=500_00,
                       payees=(HOSTING,))
    ledger = AuditLedger(Path(tempfile.mkdtemp()) / "ledger.jsonl")
    client = PayPalClient(PayPalConfig(client_id="cid",
                                       client_secret="csecret"),
                          transport=transport)
    assert client.mode == "sandbox"
    return PaymentFlow(policy=cfg,
                       state=AccountState(balance_minor=1000_00),
                       ledger=ledger, llm=MockLLMClient(),
                       paypal=client)


def propose(flow, amount="120.00", reference="INV-1042"):
    return flow.run_structured({
        "payee": "billing@northwind.example", "amount": amount,
        "memo": f"hosting invoice {reference}", "reference": reference})


class TestTwoPhaseFlow(unittest.TestCase):
    def test_pass_ends_as_order_created_not_reject(self):
        flow = make_live_flow(FakeLiveTransport())
        record = propose(flow)
        self.assertEqual(record["final_verdict"], "PASS")
        ex = record["execution"]
        self.assertEqual(ex["status"], "ORDER_CREATED")
        self.assertEqual(ex["mode"], "sandbox")
        self.assertEqual(ex["order_id"], "ORDER-1")
        self.assertEqual(ex["approval_url"], APPROVE_URL)
        self.assertIsNone(ex["capture_id"])
        card = render_card(record)
        self.assertIn("awaiting payer approval", card)
        self.assertIn(APPROVE_URL, card)
        self.assertIn("vaultpay.py capture ORDER-1", card)

    def test_pending_order_counts_against_daily_cap(self):
        flow = make_live_flow(FakeLiveTransport())
        propose(flow, "200.00", "INV-1")
        propose(flow, "200.00", "INV-2")
        self.assertEqual(flow.ledger.spent_today_minor("USD"), 400_00)
        # A third $200 would take the day to $600 > $500 cap, even
        # though nothing has been captured yet.
        record = propose(flow, "200.00", "INV-3")
        self.assertEqual(record["final_verdict"], "REJECT")
        self.assertTrue(any("Daily cap" in r for r
                            in record["policy_verdict"]["reasons"]))

    def test_capture_completes_without_double_counting(self):
        transport = FakeLiveTransport()
        flow = make_live_flow(transport)
        propose(flow)
        transport.approved = True
        capture = flow.capture_order("ORDER-1")
        self.assertEqual(capture["kind"], "capture")
        self.assertEqual(capture["order_id"], "ORDER-1")
        self.assertIsNotNone(capture["decision_ts"])
        self.assertEqual(capture["execution"]["status"], "COMPLETED")
        self.assertEqual(capture["execution"]["capture_id"], "CAP-1")
        # Append-only: decision + capture records on disk...
        self.assertEqual(len(flow.ledger.entries()), 2)
        # ...but the amount is counted once, at creation.
        self.assertEqual(flow.ledger.spent_today_minor("USD"), 120_00)
        summary = flow.ledger.summary()
        self.assertEqual(summary["executed"], 1)
        self.assertEqual(summary["awaiting_capture"], 0)

    def test_capture_is_idempotent(self):
        transport = FakeLiveTransport()
        flow = make_live_flow(transport)
        propose(flow)
        transport.approved = True
        first = flow.capture_order("ORDER-1")
        second = flow.capture_order("ORDER-1")
        self.assertEqual(first["ts"], second["ts"])
        self.assertEqual(transport.capture_calls, 1)
        self.assertEqual(len(flow.ledger.entries()), 2)

    def test_failed_capture_is_recorded_and_retryable(self):
        transport = FakeLiveTransport()
        flow = make_live_flow(transport)
        propose(flow)
        # Payer has not approved yet: PayPal refuses the capture.
        failed = flow.capture_order("ORDER-1")
        self.assertEqual(failed["execution"]["status"], "ERROR")
        self.assertIn("ORDER_NOT_APPROVED", failed["execution"]["error"])
        self.assertEqual(flow.ledger.summary()["awaiting_capture"], 1)
        # After approval the retry succeeds; the trail keeps both.
        transport.approved = True
        done = flow.capture_order("ORDER-1")
        self.assertEqual(done["execution"]["status"], "COMPLETED")
        self.assertEqual(len(flow.ledger.entries()), 3)
        self.assertEqual(flow.ledger.summary()["executed"], 1)
        self.assertEqual(flow.ledger.spent_today_minor("USD"), 120_00)

    def test_capture_unknown_order_raises_not_found(self):
        flow = make_live_flow(FakeLiveTransport())
        with self.assertRaises(CaptureError) as ctx:
            flow.capture_order("NOPE-404")
        self.assertEqual(ctx.exception.code, "not_found")

    def test_capture_while_frozen_is_refused(self):
        # The owner kill-switch halts payments: capturing while
        # frozen used to move money anyway. The order stays
        # ORDER_CREATED, PayPal is never called, and capture works
        # again once the freeze lifts.
        transport = FakeLiveTransport()
        flow = make_live_flow(transport)
        propose(flow)
        flow.policy.frozen = True
        with self.assertRaises(CaptureError) as ctx:
            flow.capture_order("ORDER-1")
        self.assertEqual(ctx.exception.code, "conflict")
        self.assertIn("FROZEN", str(ctx.exception))
        self.assertEqual(transport.capture_calls, 0)
        self.assertEqual(len(flow.ledger.entries()), 1)
        flow.policy.frozen = False
        transport.approved = True
        done = flow.capture_order("ORDER-1")
        self.assertEqual(done["execution"]["status"], "COMPLETED")

    def test_capture_of_one_phase_order_is_a_conflict(self):
        # Simulated orders complete inside propose; there is no
        # phase 2 to run for them.
        ledger = AuditLedger(Path(tempfile.mkdtemp()) / "ledger.jsonl")
        flow = PaymentFlow(
            policy=PolicyConfig(per_tx_cap_minor=250_00,
                                daily_cap_minor=500_00, payees=(HOSTING,)),
            state=AccountState(balance_minor=1000_00),
            ledger=ledger, llm=MockLLMClient(),
            paypal=SimulatedPayPalClient())
        record = propose(flow)
        order_id = record["execution"]["order_id"]
        self.assertEqual(record["execution"]["status"], "COMPLETED")
        with self.assertRaises(CaptureError) as ctx:
            flow.capture_order(order_id)
        self.assertEqual(ctx.exception.code, "conflict")


POLICY_JSON = {
    "account_balance": "1000.00",
    "allowed_currencies": ["USD"],
    "cap_currency": "USD",
    "per_tx_cap": "250.00",
    "daily_cap": "500.00",
    "frozen": False,
    "payees": [{"id": "vendor-hosting", "name": "Northwind Hosting",
                "email": "billing@northwind.example"}],
}


class TestTwoPhaseWeb(unittest.TestCase):
    """The web app over the fake-live client: card, grid, endpoint."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        policy_path = Path(self.tmp) / "policy.json"
        policy_path.write_text(json.dumps(POLICY_JSON), encoding="utf-8")
        self.transport = FakeLiveTransport()
        client = PayPalClient(PayPalConfig(client_id="cid",
                                           client_secret="csecret"),
                              transport=self.transport)
        self.app = VaultPayWebApp(
            policy_path=policy_path,
            ledger_path=Path(self.tmp) / "ledger.jsonl",
            llm=MockLLMClient(), paypal=client)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0),
                                          make_handler(self.app))
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, path, payload):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def get_json(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_order_created_card_grid_and_capture_endpoint(self):
        status, data = self.post("/api/payments", {
            "payee": "billing@northwind.example", "amount": "120.00",
            "reference": "INV-1042", "memo": "hosting invoice INV-1042"})
        self.assertEqual(status, 200)
        self.assertEqual(data["record"]["final_verdict"], "PASS")
        self.assertEqual(data["record"]["execution"]["status"],
                         "ORDER_CREATED")
        self.assertEqual(data["row"]["execution_status"], "ORDER_CREATED")
        self.assertEqual(data["row"]["stage"], "execution")
        self.assertIn("awaiting payer approval", data["row"]["reason"])
        # The card shows the approval link and the capture button.
        self.assertIn(APPROVE_URL, data["card_html"])
        self.assertIn("Complete capture", data["card_html"])
        self.assertFalse(data["simulated"])

        rows = self.get_json("/api/ledger")["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["execution_status"], "ORDER_CREATED")
        self.assertIsNone(rows[0]["capture_id"])

        # Payer approves on PayPal's side; the UI completes capture.
        self.transport.approved = True
        status, data = self.post("/api/capture", {"order_id": "ORDER-1"})
        self.assertEqual(status, 200)
        self.assertEqual(data["capture"]["kind"], "capture")
        self.assertEqual(data["record"]["execution"]["status"],
                         "COMPLETED")
        self.assertEqual(data["record"]["execution"]["capture_id"],
                         "CAP-1")
        self.assertIn("CAP-1", data["card_html"])

        # The grid reconciles the pair into the same single row, now
        # COMPLETED — while the raw ledger honestly keeps both lines.
        rows = self.get_json("/api/ledger")["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["execution_status"], "COMPLETED")
        self.assertEqual(rows[0]["capture_id"], "CAP-1")
        self.assertEqual(rows[0]["stage"], "executed")
        entries = AuditLedger(self.app.ledger_path).entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[1]["kind"], "capture")

    def test_capture_unknown_order_is_404(self):
        status, data = self.post("/api/capture", {"order_id": "NOPE"})
        self.assertEqual(status, 404)
        self.assertIn("error", data)

    def test_capture_missing_order_id_is_400(self):
        status, data = self.post("/api/capture", {})
        self.assertEqual(status, 400)
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
