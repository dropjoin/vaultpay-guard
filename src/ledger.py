"""Audit ledger — one JSON object per line, the full decision trail.

Every payment request that reaches the flow — approved or rejected —
appends exactly one *decision* record:

    ts, kind="decision", request (raw text, if any), proposal,
    guardrail, policy_verdict, checker, final_verdict, execution

Rejected proposals are recorded with ``execution: null``; executed ones
carry the PayPal order id, capture id, status and mode (``sandbox`` or
``simulated``). Live execution is two-phase (see :mod:`src.flow`), so
a decision's execution can be ``ORDER_CREATED`` — the payer still has
to approve. When capture later runs, a second record is appended:

    ts, kind="capture", order_id, decision_ts, proposal, execution

The file stays append-only; readers reconcile a capture record with
its decision record by order id.

The ledger is also the source of truth for ``spent_today``: the policy
engine's daily-cap input is read back from this file — so restarts
cannot reset the daily budget. Accounting choice: a payment counts
against the daily cap from the moment its order is CREATED, not from
capture (see :meth:`AuditLedger.spent_today_minor`).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class AuditLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: dict) -> dict:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  **record}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                # A torn write (crash mid-append) or a concurrent read
                # catching a half-written last line must never brick
                # the whole guard: skip the unparseable line and keep
                # the valid records. The raw file is untouched, so
                # the bytes stay available for forensics.
                continue
        return out

    def spent_today_minor(self, currency: str, day: str | None = None) -> int:
        """Sum of today's committed payments in `currency` (UTC day).

        A decision record counts when its execution is ``COMPLETED``
        (simulated one-phase) OR ``ORDER_CREATED`` (live phase 1).
        Counting at creation is deliberate: an approved-but-uncaptured
        order has already committed the payer, so it must consume
        daily cap — otherwise two orders could each pass the cap check
        and both be captured later, blowing through the limit.

        Capture records are NOT summed: they complete an order whose
        amount was already counted on the decision record, and adding
        them again would double-charge the cap.
        """
        day = day or _today_utc()
        total = 0
        for e in self.entries():
            if e.get("kind") == "capture":
                continue
            if not e.get("ts", "").startswith(day):
                continue
            ex = e.get("execution") or {}
            prop = e.get("proposal") or {}
            if ex.get("status") in ("COMPLETED", "ORDER_CREATED") and \
                    str(prop.get("currency", "")).upper() == currency.upper():
                try:
                    total += int(prop.get("amount_minor", 0))
                except (TypeError, ValueError):
                    # A record with a non-integer amount is not valid
                    # accounting input; skip it rather than crashing
                    # the daily-cap check for every later payment.
                    continue
        return total

    def summary(self) -> dict:
        entries = self.entries()
        decisions = [e for e in entries if e.get("kind") != "capture"]
        captures = [e for e in entries if e.get("kind") == "capture"]
        captured_ids = {
            (e.get("execution") or {}).get("order_id")
            for e in captures
            if (e.get("execution") or {}).get("status") == "COMPLETED"}
        # A completed order is counted exactly once: via its decision
        # record when execution was one-phase (status COMPLETED), or
        # via its capture record when two-phase (the decision record
        # stays ORDER_CREATED, so the two never overlap).
        executed = sum(
            1 for e in decisions
            if (e.get("execution") or {}).get("status") == "COMPLETED")
        executed += sum(
            1 for e in captures
            if (e.get("execution") or {}).get("status") == "COMPLETED")
        awaiting = sum(
            1 for e in decisions
            if (e.get("execution") or {}).get("status") == "ORDER_CREATED"
            and (e.get("execution") or {}).get("order_id")
            not in captured_ids)
        return {
            "records": len(entries),
            "decisions": len(decisions),
            "executed": executed,
            "rejected": sum(1 for e in decisions
                            if e.get("final_verdict") == "REJECT"),
            "awaiting_capture": awaiting,
        }
