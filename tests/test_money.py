"""Money math: integer minor units, exact parsing/formatting."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.money import format_amount, format_money, parse_amount  # noqa: E402


class TestMoney(unittest.TestCase):
    def test_parse_two_decimal_currency(self):
        self.assertEqual(parse_amount("120.50", "USD"), 12050)
        self.assertEqual(parse_amount("0.01", "USD"), 1)
        self.assertEqual(parse_amount("250", "USD"), 25000)

    def test_parse_zero_decimal_currency(self):
        self.assertEqual(parse_amount("500", "JPY"), 500)

    def test_parse_rejects_excess_precision(self):
        with self.assertRaises(ValueError):
            parse_amount("1.005", "USD")
        with self.assertRaises(ValueError):
            parse_amount("100.5", "JPY")

    def test_parse_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_amount("not-a-number", "USD")

    def test_parse_rejects_non_finite_as_value_error(self):
        # "Infinity" used to escape as OverflowError and "NaN" with a
        # misleading precision message; both are plain ValueErrors now,
        # so callers map them to a 422 instead of crashing.
        for text in ("Infinity", "-Infinity", "NaN", "inf"):
            with self.assertRaises(ValueError, msg=text):
                parse_amount(text, "USD")

    def test_parse_rejects_absurd_magnitude_as_value_error(self):
        # A compact exponent ("1e999999") used to escape as
        # decimal.Overflow; it is a ValueError now.
        for text in ("1e999999", "1e100000000", "-1e999999"):
            with self.assertRaises(ValueError, msg=text):
                parse_amount(text, "USD")

    def test_parse_accepts_large_but_sane_amounts(self):
        self.assertEqual(parse_amount("1000000.00", "USD"), 100_000_000)
        self.assertEqual(parse_amount("1e6", "USD"), 100_000_000)

    def test_format_roundtrip(self):
        self.assertEqual(format_amount(12050, "USD"), "120.50")
        self.assertEqual(format_amount(1, "USD"), "0.01")
        self.assertEqual(format_amount(500, "JPY"), "500")
        self.assertEqual(format_money(25000, "USD"), "USD 250.00")
        for minor in (0, 1, 99, 100, 123456789):
            self.assertEqual(parse_amount(format_amount(minor, "USD"), "USD"),
                             minor)


if __name__ == "__main__":
    unittest.main()
