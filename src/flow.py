"""End-to-end payment flow — the pipeline every request travels.

    request text / structured input
        -> PROPOSER drafts a PaymentProposal (AI role A)
        -> GUARDRAIL scans all untrusted text (deterministic)
        -> POLICY ENGINE evaluates caps / allowlist / balance (code)
        -> CHECKER reviews (AI role B, independent, tighten-only)
        -> EXECUTE via PayPal Orders (sandbox or clearly-labeled
           simulated mode)
        -> LEDGER appends the full decision record, pass or fail

Execution is TWO-PHASE against the real Orders v2 API, because that
is how PayPal works: ``create_order`` returns a CREATED order, the
payer must approve it, and only then may ``capture_order`` run. A
policy PASS therefore ends phase 1 as execution status
``ORDER_CREATED`` (verdict PASS — the guard approved the payment;
capture is still pending), and :meth:`PaymentFlow.capture_order`
runs phase 2 later, appending its own ``kind="capture"`` audit
record. In simulated mode there is no payer, so the twin client
captures immediately and nothing changes for offline runs.

Nothing moves unless every layer passes. A failure at any layer —
including a model error or an unparseable model answer — ends the
flow as a REJECT with the reason recorded. That is the product.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import agents
from .guardrail import merge, scan_untrusted_text
from .ledger import AuditLedger
from .llm import LLMClient
from .money import format_money
from .paypal_client import PayPalError, approval_url
from .policy import AccountState, PaymentProposal, PolicyConfig, evaluate


class CaptureError(Exception):
    """A capture was requested for an order that cannot be captured.

    ``code`` is ``"not_found"`` when no decision record in the ledger
    carries the order id, ``"conflict"`` when the order exists but is
    not awaiting capture.
    """

    def __init__(self, message: str, code: str = "conflict"):
        super().__init__(message)
        self.code = code


@dataclass
class PaymentFlow:
    policy: PolicyConfig
    state: AccountState
    ledger: AuditLedger
    llm: LLMClient | None
    paypal: object  # PayPalClient | SimulatedPayPalClient (same interface)

    # -- entry points ---------------------------------------------------
    def run_text(self, request_text: str) -> dict:
        """Natural-language request: the proposer extracts the fields."""
        if self.llm is None:
            raise agents.ProposalError(
                "Natural-language requests need a proposer model; "
                "pass structured input instead.")
        proposal = agents.draft_proposal(self.llm, request_text)
        return self._run(proposal, request_text=request_text)

    def run_structured(self, data: dict) -> dict:
        """Structured input (form/API): no extraction step needed."""
        proposal = agents.proposal_from_mapping(data)
        return self._run(proposal, request_text=str(data.get("memo", "")))

    # -- the pipeline ---------------------------------------------------
    def _run(self, proposal: PaymentProposal, request_text: str) -> dict:
        # Refresh spent-today from the ledger so the daily cap survives
        # restarts and reflects every executed payment so far today.
        self.state.spent_today_minor = self.ledger.spent_today_minor(
            self.policy.cap_currency)
        if proposal.currency.upper() == self.policy.cap_currency:
            self.state.currency = self.policy.cap_currency

        guardrail = merge(scan_untrusted_text(request_text),
                          scan_untrusted_text(proposal.memo),
                          scan_untrusted_text(proposal.rationale))
        det = evaluate(self.policy, self.state, proposal)
        outcome = agents.run_checker(self.llm, proposal, self.policy,
                                     self.state, det, guardrail,
                                     request_text=request_text)

        execution = None
        final_verdict = outcome.final_verdict
        if final_verdict == "PASS":
            execution = self._execute(proposal)
            # COMPLETED (simulated one-phase) and ORDER_CREATED (live
            # phase 1) both keep the PASS: the guard approved the
            # payment. Anything else means execution itself failed.
            if execution["status"] not in ("COMPLETED", "ORDER_CREATED"):
                final_verdict = "REJECT"

        record = {
            "kind": "decision",
            "request": request_text,
            "proposal": {
                "payee": proposal.payee,
                "payee_id": det.facts.get("payee_id"),
                "amount_minor": proposal.amount_minor,
                "amount": proposal.amount_str(),
                "currency": proposal.currency.upper(),
                "reference": proposal.reference,
                "memo": proposal.memo,
                "rationale": proposal.rationale,
            },
            "guardrail": guardrail.to_dict(),
            "policy_verdict": {"verdict": det.verdict,
                               "reasons": det.reasons, "facts": det.facts},
            "checker": {"source": outcome.source,
                        "llm_verdict": outcome.llm_verdict,
                        "reasons": outcome.reasons},
            "final_verdict": final_verdict,
            "execution": execution,
        }
        return self.ledger.append(record)

    def _execute(self, proposal: PaymentProposal) -> dict:
        """Phase 1 of execution: create the PayPal order.

        Live modes stop here — the payer still has to approve before
        capture is even possible (see :meth:`capture_order`). The
        simulated client has no payer, so it captures immediately,
        exactly as before.
        """
        payee = self.policy.find_payee(proposal.payee)
        mode = getattr(self.paypal, "mode", "unknown")
        try:
            order = self.paypal.create_order(
                amount_minor=proposal.amount_minor,
                currency=proposal.currency,
                payee_email=payee.email if payee else "",
                reference=proposal.reference,
                description=(proposal.rationale or proposal.memo)[:127],
            )
        except PayPalError as exc:
            return {"status": "ERROR", "mode": mode, "error": str(exc)}
        if mode != "simulated":
            return {
                "status": "ORDER_CREATED",
                "mode": mode,
                "order_id": order.get("id"),
                "capture_id": None,
                "approval_url": approval_url(order),
                "amount": proposal.amount_str(),
            }
        try:
            captured = self.paypal.capture_order(order["id"])
            captures = (captured.get("purchase_units", [{}])[0]
                        .get("payments", {}).get("captures", []))
            return {
                "status": captured.get("status", "UNKNOWN"),
                "mode": mode,
                "order_id": captured.get("id"),
                "capture_id": captures[0]["id"] if captures else None,
                "amount": proposal.amount_str(),
            }
        except PayPalError as exc:
            return {"status": "ERROR", "mode": mode, "error": str(exc)}

    # -- phase 2 ----------------------------------------------------------
    def capture_order(self, order_id: str) -> dict:
        """Capture an ORDER_CREATED order after payer approval.

        The ledger is append-only, so the outcome is appended as a
        ``kind="capture"`` record referencing the original decision
        (``decision_ts``); readers reconcile the pair. Idempotent: if
        a COMPLETED capture already exists for the order, that record
        is returned and PayPal is not called again. A failed capture
        (e.g. the payer has not approved yet) is still recorded —
        status ``ERROR`` with the PayPal message — and the order
        stays ORDER_CREATED, ready to retry after approval.
        """
        decision = None
        existing = None
        for entry in self.ledger.entries():
            ex = entry.get("execution") or {}
            if entry.get("kind") == "capture":
                if ex.get("order_id") == order_id or \
                        entry.get("order_id") == order_id:
                    existing = entry
                continue
            if ex.get("order_id") == order_id:
                decision = entry
        if existing is not None and \
                (existing.get("execution") or {}).get("status") == "COMPLETED":
            return existing
        if decision is None:
            raise CaptureError(
                f"No order {order_id} found in the audit ledger.",
                code="not_found")
        status = (decision.get("execution") or {}).get("status")
        if status != "ORDER_CREATED":
            raise CaptureError(
                f"Order {order_id} is {status or 'not executed'}, "
                "not awaiting capture.")
        prop = decision.get("proposal") or {}
        mode = getattr(self.paypal, "mode", "unknown")
        try:
            captured = self.paypal.capture_order(order_id)
            captures = (captured.get("purchase_units", [{}])[0]
                        .get("payments", {}).get("captures", []))
            execution = {
                "status": captured.get("status", "UNKNOWN"),
                "mode": mode,
                "order_id": captured.get("id", order_id),
                "capture_id": captures[0]["id"] if captures else None,
                "amount": prop.get("amount", ""),
            }
        except PayPalError as exc:
            execution = {"status": "ERROR", "mode": mode,
                         "order_id": order_id, "capture_id": None,
                         "amount": prop.get("amount", ""),
                         "error": str(exc)}
        record = {
            "kind": "capture",
            "order_id": order_id,
            "decision_ts": decision.get("ts"),
            "proposal": {key: prop.get(key) for key in
                         ("payee", "payee_id", "amount_minor", "amount",
                          "currency", "reference")},
            "execution": execution,
        }
        return self.ledger.append(record)


def render_card(record: dict) -> str:
    """Plain-language decision card (local template, no model needed)."""
    p = record["proposal"]
    facts = record["policy_verdict"]["facts"]
    lines = [
        "┌─ VAULTPAY GUARD — DECISION CARD ──────────────────────",
        f"│ Verdict   : {record['final_verdict']}",
        f"│ Payee     : {p['payee']}"
        + (f" (allowlist id: {p['payee_id']})" if p.get("payee_id") else ""),
        f"│ Amount    : {p['amount']}",
        f"│ Policy    : per-tx {facts['per_tx_cap']} · daily "
        f"{facts['daily_cap']} · headroom {facts['daily_headroom']}",
        f"│ Balance   : {facts['balance']} · frozen={facts['frozen']}",
        "│ Reasons   :",
    ]
    for reason in record["checker"]["reasons"]:
        lines.append(f"│   - {reason}")
    lines.append(f"│ Guardrail : {record['guardrail']['summary']}")
    lines.append(f"│ Checked by: {record['checker']['source']}")
    ex = record.get("execution")
    if ex:
        if ex["status"] == "COMPLETED":
            lines.append(f"│ PayPal    : order {ex['order_id']} · capture "
                         f"{ex['capture_id']} · mode={ex['mode']}")
        elif ex["status"] == "ORDER_CREATED":
            lines.append(f"│ PayPal    : order {ex['order_id']} · CREATED, "
                         f"awaiting payer approval · mode={ex['mode']}")
            if ex.get("approval_url"):
                lines.append(f"│ Approve   : {ex['approval_url']}")
            lines.append(f"│ Next      : after payer approval, run: "
                         f"vaultpay.py capture {ex['order_id']}")
        else:
            lines.append(f"│ PayPal    : EXECUTION FAILED — "
                         f"{ex.get('error', ex['status'])}")
    lines.append("└───────────────────────────────────────────────────────")
    return "\n".join(lines)
