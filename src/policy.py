"""Deterministic policy engine — ported from VaultMind's policy mirror.

In VaultMind this module mirrored an on-chain vault's ``require()``
chain so a verdict could be explained *before* anything was signed.
VaultPay Guard keeps the same architecture with fiat rules in place of
chain rules. The engine is plain code, never a model, and it is the
floor every other layer stands on:

    account not frozen                 -> owner kill-switch is off
    amount > 0                         -> amount positive
    payee on the allowlist             -> recipient allowlisted
    currency allowed                   -> currency has a policy
    amount <= per-transaction cap      -> per-tx cap (cap currency only)
    spent_today + amount <= daily cap  -> daily cap (cap currency only)
    amount <= account balance          -> funds actually present

Fail-closed choices, deliberately:
- Caps are denominated in one ``cap_currency``. A payment in a different
  (otherwise allowed) currency is REJECTED: with no trusted FX source
  the engine refuses to guess what the payment is "worth".
- A policy that fails to load is not a policy: callers must construct
  :class:`PolicyConfig` explicitly, and an empty allowlist allows
  nobody.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .money import format_money, parse_amount


@dataclass(frozen=True)
class Payee:
    id: str
    name: str
    email: str

    def matches(self, candidate: str) -> bool:
        c = candidate.strip().lower()
        return c in (self.id.lower(), self.email.lower(), self.name.lower())


@dataclass
class PolicyConfig:
    per_tx_cap_minor: int
    daily_cap_minor: int
    cap_currency: str = "USD"
    allowed_currencies: tuple[str, ...] = ("USD",)
    payees: tuple[Payee, ...] = ()
    frozen: bool = False

    def find_payee(self, candidate: str) -> Payee | None:
        for payee in self.payees:
            if payee.matches(candidate):
                return payee
        return None

    @classmethod
    def from_json(cls, data: dict) -> "PolicyConfig":
        cap_currency = str(data.get("cap_currency", "USD")).upper()
        return cls(
            per_tx_cap_minor=parse_amount(str(data["per_tx_cap"]), cap_currency),
            daily_cap_minor=parse_amount(str(data["daily_cap"]), cap_currency),
            cap_currency=cap_currency,
            allowed_currencies=tuple(
                str(c).upper() for c in data.get("allowed_currencies", [cap_currency])),
            payees=tuple(
                Payee(id=str(p["id"]), name=str(p.get("name", p["id"])),
                      email=str(p.get("email", "")))
                for p in data.get("payees", [])),
            frozen=bool(data.get("frozen", False)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "PolicyConfig":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class AccountState:
    """Live-ish account facts the engine checks against.

    ``spent_today_minor`` is derived from the audit ledger (sum of
    today's executed payments in the cap currency); ``balance_minor``
    is the sandbox account balance as last read/seeded. Both are
    inputs, not guesses — the flow layer supplies them.
    """
    balance_minor: int
    spent_today_minor: int = 0
    currency: str = "USD"


@dataclass
class PaymentProposal:
    payee: str            # payee id, name, or email as the requester stated it
    amount_minor: int     # minor units of `currency`
    currency: str = "USD"
    memo: str = ""        # untrusted free text (guardrail input)
    reference: str = ""   # e.g. invoice number, if any
    rationale: str = ""   # proposer's stated reason (also untrusted-ish)

    def amount_str(self) -> str:
        return format_money(self.amount_minor, self.currency)


@dataclass
class PolicyVerdict:
    verdict: str  # "PASS" or "REJECT"
    reasons: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


def evaluate(policy: PolicyConfig, state: AccountState,
             proposal: PaymentProposal) -> PolicyVerdict:
    """Run every deterministic check, collecting *all* failures."""
    reasons: list[str] = []
    currency = proposal.currency.upper()
    payee = policy.find_payee(proposal.payee)

    if policy.frozen:
        reasons.append("Account is FROZEN by the owner — agent payments "
                       "are halted.")
    if proposal.amount_minor <= 0:
        reasons.append("Amount must be greater than zero.")
    if payee is None:
        reasons.append(f"Payee {proposal.payee!r} is NOT on the allowlist.")
    if currency not in policy.allowed_currencies:
        reasons.append(f"Currency {currency} is not allowed by policy "
                       f"(allowed: {', '.join(policy.allowed_currencies)}).")
    elif currency != policy.cap_currency:
        # Fail closed: caps exist only in the cap currency and there is
        # no trusted FX source to convert with.
        reasons.append(
            f"Currency {currency} is allowed but caps are denominated in "
            f"{policy.cap_currency} and no FX rate is configured — "
            "failing closed.")
    else:
        if proposal.amount_minor > policy.per_tx_cap_minor:
            reasons.append(
                f"Amount {proposal.amount_str()} exceeds the "
                f"per-transaction cap of "
                f"{format_money(policy.per_tx_cap_minor, policy.cap_currency)}.")
        if state.spent_today_minor + proposal.amount_minor > policy.daily_cap_minor:
            reasons.append(
                f"Daily cap exceeded: "
                f"{format_money(state.spent_today_minor, policy.cap_currency)} "
                f"already spent today + {proposal.amount_str()} proposed > "
                f"daily cap "
                f"{format_money(policy.daily_cap_minor, policy.cap_currency)}.")
    if proposal.amount_minor > state.balance_minor:
        reasons.append(
            f"Amount exceeds the account balance "
            f"({format_money(state.balance_minor, state.currency)} available).")

    facts = {
        "per_tx_cap": format_money(policy.per_tx_cap_minor, policy.cap_currency),
        "daily_cap": format_money(policy.daily_cap_minor, policy.cap_currency),
        "spent_today": format_money(state.spent_today_minor, policy.cap_currency),
        "daily_headroom": format_money(
            max(policy.daily_cap_minor - state.spent_today_minor, 0),
            policy.cap_currency),
        "balance": format_money(state.balance_minor, state.currency),
        "frozen": policy.frozen,
        "payee_allowlisted": payee is not None,
        "payee_id": payee.id if payee else None,
        "currency": currency,
    }
    return PolicyVerdict(
        verdict="PASS" if not reasons else "REJECT",
        reasons=reasons or ["All deterministic policy checks pass."],
        facts=facts,
    )
