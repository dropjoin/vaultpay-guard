"""Audit ledger: append-only reads must survive damaged lines.

A torn write (crash mid-append) or a concurrent read catching a
half-written last line used to crash entries() — and with it every
payment, because the daily-cap check reads the ledger first. The
ledger now skips unparseable lines (the raw file is untouched) and
skips records whose amount is not an integer.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ledger import AuditLedger  # noqa: E402

DAY = "2026-11-12"


def _decision(amount_minor, status="COMPLETED"):
    return ('{"kind": "decision", "ts": "%sT06:00:00+00:00", '
            '"execution": {"status": "%s"}, "proposal": '
            '{"currency": "USD", "amount_minor": %s}}'
            % (DAY, status, amount_minor))


class TestLedgerRobustness(unittest.TestCase):
    def _ledger_with(self, text: str) -> AuditLedger:
        path = Path(tempfile.mkdtemp()) / "ledger.jsonl"
        path.write_text(text, encoding="utf-8")
        return AuditLedger(path)

    def test_corrupt_line_is_skipped_not_fatal(self):
        ledger = self._ledger_with(_decision(1000) + "\n{\"corrupt\n"
                                   + _decision(2000) + "\n")
        entries = ledger.entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual(ledger.spent_today_minor("USD", day=DAY), 3000)
        self.assertEqual(ledger.summary()["records"], 2)

    def test_blank_lines_are_skipped(self):
        ledger = self._ledger_with("\n" + _decision(1000) + "\n\n")
        self.assertEqual(len(ledger.entries()), 1)

    def test_non_integer_amount_is_skipped_in_spent_today(self):
        ledger = self._ledger_with(_decision('"abc"') + "\n"
                                   + _decision(1000) + "\n")
        self.assertEqual(ledger.spent_today_minor("USD", day=DAY), 1000)

    def test_missing_file_reads_empty(self):
        ledger = AuditLedger(Path(tempfile.mkdtemp()) / "nope.jsonl")
        self.assertEqual(ledger.entries(), [])
        self.assertEqual(ledger.spent_today_minor("USD", day=DAY), 0)


if __name__ == "__main__":
    unittest.main()
