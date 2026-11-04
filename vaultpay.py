#!/usr/bin/env python3
"""VaultPay Guard CLI — a policy-guarded AI payment agent on PayPal.

Commands:
  propose  Run one payment request through the full pipeline:
           proposer -> guardrail -> policy engine -> checker ->
           PayPal create order -> audit ledger. Against the real
           Orders API the order is left ORDER_CREATED (the payer
           still has to approve it); in simulated mode it is captured
           immediately.
  capture  Phase 2 for a live order: capture an ORDER_CREATED order
           after payer approval, appending a capture record to the
           audit ledger.
  policy   Show the active policy (caps, allowlist) and today's
           spend/headroom read back from the audit ledger.
  ledger   Summary of the audit ledger.

Backends are chosen by environment (see .env.example): with no LLM
key the deterministic MOCK model is used; with no PayPal credentials
the PayPal client runs in clearly-labeled SIMULATED mode. The guard
pipeline behaves identically either way.

Exit codes: 0 = ok / PASS · 3 = proposal REJECTED · 1 = runtime error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as cfg  # noqa: E402
from src.agents import ProposalError  # noqa: E402
from src.flow import CaptureError, PaymentFlow, render_card  # noqa: E402
from src.ledger import AuditLedger  # noqa: E402
from src.llm import build_llm_client  # noqa: E402
from src.paypal_client import build_paypal_client  # noqa: E402
from src.policy import AccountState, PolicyConfig  # noqa: E402


def _make_flow() -> PaymentFlow:
    policy = PolicyConfig.from_file(cfg.policy_path())
    ledger = AuditLedger(cfg.ledger_path())
    state = AccountState(
        balance_minor=cfg.account_balance_minor(policy.cap_currency),
        spent_today_minor=ledger.spent_today_minor(policy.cap_currency),
        currency=policy.cap_currency)
    return PaymentFlow(policy=policy, state=state, ledger=ledger,
                       llm=build_llm_client(), paypal=build_paypal_client())


def cmd_propose(args) -> int:
    flow = _make_flow()
    print(f"[backends] llm={flow.llm.name} · paypal mode="
          f"{getattr(flow.paypal, 'mode', 'unknown')}")
    try:
        if args.text:
            record = flow.run_text(args.text)
        else:
            record = flow.run_structured({
                "payee": args.payee, "amount": args.amount,
                "currency": args.currency, "memo": args.memo or "",
                "reference": args.reference or "",
            })
    except ProposalError as exc:
        print(f"[proposer FAILED] {exc}")
        return 1
    print(render_card(record))
    return 0 if record["final_verdict"] == "PASS" else 3


def cmd_capture(args) -> int:
    flow = _make_flow()
    print(f"[backends] llm={flow.llm.name} · paypal mode="
          f"{getattr(flow.paypal, 'mode', 'unknown')}")
    try:
        record = flow.capture_order(args.order_id)
    except CaptureError as exc:
        print(f"[capture] {exc}")
        return 1
    ex = record["execution"]
    if ex["status"] == "COMPLETED":
        print(f"[capture] order {ex['order_id']} · capture "
              f"{ex['capture_id']} · COMPLETED · mode={ex['mode']} · "
              "recorded in the audit ledger")
        return 0
    print(f"[capture] order {ex['order_id']} · {ex['status']}"
          + (f" — {ex['error']}" if ex.get("error") else "")
          + " · attempt recorded in the audit ledger; the order stays "
            "ORDER_CREATED and capture can be retried after approval")
    return 1


def cmd_policy(_args) -> int:
    policy = PolicyConfig.from_file(cfg.policy_path())
    ledger = AuditLedger(cfg.ledger_path())
    from src.money import format_money
    spent = ledger.spent_today_minor(policy.cap_currency)
    print(f"Policy ({cfg.policy_path()})")
    print(f"  frozen         : {policy.frozen}")
    print(f"  cap currency   : {policy.cap_currency}")
    print(f"  per-tx cap     : {format_money(policy.per_tx_cap_minor, policy.cap_currency)}")
    print(f"  daily cap      : {format_money(policy.daily_cap_minor, policy.cap_currency)}")
    print(f"  spent today    : {format_money(spent, policy.cap_currency)} "
          f"(from audit ledger)")
    print(f"  balance (seed) : {format_money(cfg.account_balance_minor(policy.cap_currency), policy.cap_currency)}")
    print(f"  currencies     : {', '.join(policy.allowed_currencies)}")
    print("  allowlist      :")
    for p in policy.payees:
        print(f"    - {p.id}: {p.name} <{p.email}>")
    return 0


def cmd_ledger(_args) -> int:
    ledger = AuditLedger(cfg.ledger_path())
    s = ledger.summary()
    print(f"Audit ledger {cfg.ledger_path()}: {s['records']} records, "
          f"{s['executed']} executed, {s['rejected']} rejected, "
          f"{s['awaiting_capture']} awaiting payer approval/capture.")
    for e in ledger.entries()[-5:]:
        p = e.get("proposal", {})
        print(f"  {e.get('ts')} {e.get('final_verdict')} "
              f"{p.get('amount', '')} -> {p.get('payee', '')}")
    return 0


def main() -> int:
    cfg.load_env()
    parser = argparse.ArgumentParser(prog="vaultpay", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prop = sub.add_parser("propose", help="run a payment request through the guard pipeline")
    p_prop.add_argument("--text", help="natural-language payment request")
    p_prop.add_argument("--payee", help="payee id/name/email (structured mode)")
    p_prop.add_argument("--amount", help="decimal amount in major units (structured mode)")
    p_prop.add_argument("--currency", default="USD")
    p_prop.add_argument("--memo", default="", help="requester memo (untrusted)")
    p_prop.add_argument("--reference", default="", help="invoice/reference id")
    p_prop.set_defaults(fn=cmd_propose)

    p_cap = sub.add_parser(
        "capture", help="capture an ORDER_CREATED order after payer approval")
    p_cap.add_argument("order_id", help="PayPal order id from a propose run")
    p_cap.set_defaults(fn=cmd_capture)

    p_pol = sub.add_parser("policy", help="show policy + today's spend")
    p_pol.set_defaults(fn=cmd_policy)

    p_led = sub.add_parser("ledger", help="audit ledger summary")
    p_led.set_defaults(fn=cmd_ledger)

    args = parser.parse_args()
    if args.command == "propose" and not args.text and \
            not (args.payee and args.amount):
        parser.error("propose needs --text, or --payee and --amount")
    try:
        return args.fn(args)
    except SystemExit:
        raise
    except Exception as exc:  # CLI surface: one clean line, no traceback
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
