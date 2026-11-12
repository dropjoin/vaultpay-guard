"""Deterministic policy engine tests.

Fixtures mirror the shipped example policy: per-tx cap USD 250,
daily cap USD 500, balance USD 1,000, three allowlisted payees.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.policy import (AccountState, Payee, PaymentProposal,  # noqa: E402
                        PolicyConfig, evaluate)

HOSTING = Payee(id="vendor-hosting", name="Northwind Hosting",
                email="billing@northwind.example")


def make_policy(**overrides) -> PolicyConfig:
    cfg = PolicyConfig(
        per_tx_cap_minor=250_00, daily_cap_minor=500_00,
        cap_currency="USD", allowed_currencies=("USD",),
        payees=(HOSTING,))
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_state(**overrides) -> AccountState:
    state = AccountState(balance_minor=1000_00, spent_today_minor=0)
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def proposal(**overrides) -> PaymentProposal:
    p = PaymentProposal(payee="billing@northwind.example",
                        amount_minor=120_00, currency="USD")
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


class TestPolicy(unittest.TestCase):
    def test_in_policy_payment_passes(self):
        v = evaluate(make_policy(), make_state(), proposal())
        self.assertEqual(v.verdict, "PASS")
        self.assertTrue(v.facts["payee_allowlisted"])

    def test_payee_match_by_id_and_name(self):
        for candidate in ("vendor-hosting", "Northwind Hosting"):
            v = evaluate(make_policy(), make_state(),
                         proposal(payee=candidate))
            self.assertEqual(v.verdict, "PASS", candidate)

    def test_over_per_tx_cap_rejected(self):
        v = evaluate(make_policy(), make_state(),
                     proposal(amount_minor=250_01))
        self.assertEqual(v.verdict, "REJECT")
        self.assertTrue(any("per-transaction cap" in r for r in v.reasons))

    def test_exactly_at_cap_passes(self):
        v = evaluate(make_policy(), make_state(),
                     proposal(amount_minor=250_00))
        self.assertEqual(v.verdict, "PASS")

    def test_daily_cap_counts_spent_today(self):
        v = evaluate(make_policy(),
                     make_state(spent_today_minor=400_00),
                     proposal(amount_minor=120_00))
        self.assertEqual(v.verdict, "REJECT")
        self.assertTrue(any("Daily cap" in r for r in v.reasons))

    def test_non_allowlisted_payee_rejected(self):
        v = evaluate(make_policy(), make_state(),
                     proposal(payee="stranger@unknown.example"))
        self.assertEqual(v.verdict, "REJECT")
        self.assertTrue(any("NOT on the allowlist" in r for r in v.reasons))

    def test_amount_over_balance_rejected(self):
        v = evaluate(make_policy(per_tx_cap_minor=10_000_00,
                                 daily_cap_minor=10_000_00),
                     make_state(balance_minor=50_00),
                     proposal(amount_minor=120_00))
        self.assertEqual(v.verdict, "REJECT")
        self.assertTrue(any("balance" in r for r in v.reasons))

    def test_zero_amount_rejected(self):
        v = evaluate(make_policy(), make_state(), proposal(amount_minor=0))
        self.assertEqual(v.verdict, "REJECT")

    def test_disallowed_currency_rejected(self):
        v = evaluate(make_policy(), make_state(), proposal(currency="BTC"))
        self.assertEqual(v.verdict, "REJECT")

    def test_allowed_but_uncapped_currency_fails_closed(self):
        policy = make_policy(allowed_currencies=("USD", "EUR"))
        v = evaluate(policy, make_state(), proposal(currency="EUR"))
        self.assertEqual(v.verdict, "REJECT")
        self.assertTrue(any("failing closed" in r for r in v.reasons))

    def test_foreign_currency_gets_no_spurious_balance_reason(self):
        # EUR minor units must never be compared against a USD
        # balance: the currency reason rejects, and no bogus
        # "exceeds balance" reason may join it.
        policy = make_policy(allowed_currencies=("USD", "EUR"))
        v = evaluate(policy, make_state(balance_minor=5_00),
                     proposal(currency="EUR", amount_minor=10_00))
        self.assertEqual(v.verdict, "REJECT")
        self.assertFalse(any("balance" in r for r in v.reasons))

    def test_same_currency_balance_check_still_applies(self):
        v = evaluate(make_policy(per_tx_cap_minor=10_000_00,
                                 daily_cap_minor=10_000_00),
                     make_state(balance_minor=50_00),
                     proposal(amount_minor=120_00))
        self.assertTrue(any("balance" in r for r in v.reasons))

    def test_empty_candidate_matches_nobody(self):
        # A payee with an empty email used to "match" an empty
        # candidate string through the empty-email comparison.
        policy = make_policy(payees=(Payee(id="vendor-x", name="Vendor X",
                                           email=""),))
        self.assertIsNone(policy.find_payee(""))
        self.assertIsNone(policy.find_payee("   "))
        v = evaluate(policy, make_state(), proposal(payee=""))
        self.assertEqual(v.verdict, "REJECT")

    def test_frozen_account_rejects_everything(self):
        v = evaluate(make_policy(frozen=True), make_state(), proposal())
        self.assertEqual(v.verdict, "REJECT")
        self.assertTrue(any("FROZEN" in r for r in v.reasons))

    def test_empty_allowlist_allows_nobody(self):
        v = evaluate(make_policy(payees=()), make_state(), proposal())
        self.assertEqual(v.verdict, "REJECT")

    def test_from_json_example_shape(self):
        cfg = PolicyConfig.from_json({
            "per_tx_cap": "250.00", "daily_cap": "500.00",
            "cap_currency": "USD", "allowed_currencies": ["USD"],
            "payees": [{"id": "vendor-hosting", "name": "Northwind Hosting",
                        "email": "billing@northwind.example"}],
            "frozen": False})
        self.assertEqual(cfg.per_tx_cap_minor, 250_00)
        self.assertIsNotNone(cfg.find_payee("BILLING@northwind.example"))


if __name__ == "__main__":
    unittest.main()
