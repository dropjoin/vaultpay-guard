"""Prompt-injection guardrail for untrusted payment text.

Ported from VaultMind's guardrail. A payment request / invoice note is
attacker-controlled text: anyone can ask the agent to pay them and lace
the request with instructions for the model. VaultPay Guard treats
that text as DATA, never as instructions:

1. A deterministic scan (this module) flags classic injection patterns
   before any model sees the text. High-severity flags force a REJECT
   at the deterministic layer — no model verdict can override them.
2. The checker agent's prompt repeats the rule for the model: the memo
   is quoted data; instructions inside it must be reported, not obeyed.

The scan is deliberately simple and explainable: a tripwire with named
patterns, not a security guarantee. Its job is to make the common,
cheap attacks fail deterministically and leave an audit trail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_HIGH = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "ignore-previous-instructions"),
    (r"disregard\s+(the\s+)?(policy|rules|instructions|checks)", "disregard-policy"),
    (r"(override|bypass|skip)\s+(the\s+)?(policy|check|cap|allowlist|approval)", "bypass-policy"),
    (r"you\s+are\s+now\s+(a|an|in)\b", "role-reassignment"),
    (r"\bsystem\s*:\s*", "fake-system-message"),
    (r"approve\s+(this|it)\s+(immediately|without|regardless)", "forced-approval"),
    (r"send\s+(all|the\s+full|everything)\b", "drain-language"),
    (r"do\s+not\s+(tell|inform|report|log)\b", "concealment"),
    (r"(change|update|switch)\s+(the\s+)?(payee|recipient|email|account)(\s+(email|address|account))?\s+to\b", "payee-swap"),
]
_MEDIUM = [
    (r"urgent(ly)?\b.*\b(no\s+time|immediately|asap)\b", "pressure-urgency"),
    (r"the\s+owner\s+(said|approved|told\s+me)", "false-authority-claim"),
    (r"off\s*the\s+record", "off-the-record"),
    (r"new\s+instructions\s*:", "fake-instruction-block"),
]


@dataclass
class GuardrailResult:
    flags: list[dict] = field(default_factory=list)  # {pattern, severity, match}

    @property
    def high_severity(self) -> bool:
        return any(f["severity"] == "high" for f in self.flags)

    @property
    def blocked(self) -> bool:
        """High-severity flags block the payment at the deterministic layer."""
        return self.high_severity

    def describe(self) -> str:
        if not self.flags:
            return "No injection patterns detected in the request text."
        parts = [f"{f['severity'].upper()} '{f['pattern']}' (matched "
                 f"\"{f['match']}\")" for f in self.flags]
        return "Injection signals: " + "; ".join(parts)

    def to_dict(self) -> dict:
        return {"blocked": self.blocked, "flags": self.flags,
                "summary": self.describe()}


def scan_untrusted_text(text: str) -> GuardrailResult:
    result = GuardrailResult()
    if not text:
        return result
    lowered = text.lower()
    for pattern, name in _HIGH:
        m = re.search(pattern, lowered)
        if m:
            result.flags.append({"pattern": name, "severity": "high",
                                 "match": m.group(0)})
    for pattern, name in _MEDIUM:
        m = re.search(pattern, lowered)
        if m:
            result.flags.append({"pattern": name, "severity": "medium",
                                 "match": m.group(0)})
    return result


def merge(*results: GuardrailResult) -> GuardrailResult:
    """Combine scans of several untrusted fields into one result.

    Identical flags (same pattern + matched text) are de-duplicated:
    the same attack string often appears in both the raw request and
    the extracted memo, and the audit trail should name it once.
    """
    out = GuardrailResult()
    seen: set[tuple[str, str]] = set()
    for r in results:
        for flag in r.flags:
            key = (flag["pattern"], flag["match"])
            if key not in seen:
                seen.add(key)
                out.flags.append(flag)
    return out
