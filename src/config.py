"""Environment + file configuration.

A tiny ``.env`` loader (standard library only) and path resolution
for the policy file and the audit ledger. Secrets are read from the
environment by the clients that need them; this module never stores
or prints them.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ (no override)."""
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def policy_path() -> Path:
    # POLICY_PATH is the deployment-facing name (Render, Docker, the
    # web layer); VAULTPAY_POLICY is the original local name. Both work.
    return Path(os.environ.get("POLICY_PATH")
                or os.environ.get("VAULTPAY_POLICY")
                or (ROOT / "config" / "policy.example.json"))


def ledger_path() -> Path:
    return Path(os.environ.get("LEDGER_PATH")
                or os.environ.get("VAULTPAY_LEDGER")
                or (ROOT / "data" / "audit-ledger.jsonl"))


def account_balance_minor(policy_currency: str,
                          path: str | Path | None = None) -> int:
    """Seed account balance from the policy file's ``account_balance``.

    In a later step this is replaced by a live sandbox balance read;
    for the offline core it comes from the same JSON the operator
    edits, so the number the engine checks against is never a guess
    buried in code.
    """
    import json

    from .money import parse_amount
    data = json.loads((Path(path) if path else policy_path())
                      .read_text(encoding="utf-8"))
    return parse_amount(str(data.get("account_balance", "0")), policy_currency)
