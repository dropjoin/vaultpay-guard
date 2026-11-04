"""Money handling — integer minor units only, never floats.

Every amount inside VaultPay Guard is an ``int`` of minor units (cents
for USD/EUR/GBP, whole yen for JPY, ...) plus an ISO currency code.
Parsing goes through :class:`decimal.Decimal` so a value like ``0.1``
can never pick up binary float error on its way into a payment
decision — the same discipline VaultMind applied to wei amounts,
adapted to fiat.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation

# ISO 4217 minor-unit exponents for the currencies the demo touches.
# Unknown currencies default to 2 (the overwhelmingly common case).
MINOR_UNITS: dict[str, int] = {
    "USD": 2, "EUR": 2, "GBP": 2, "SGD": 2, "AUD": 2, "CAD": 2,
    "IDR": 2, "JPY": 0, "KRW": 0,
}


def minor_units(currency: str) -> int:
    return MINOR_UNITS.get(currency.upper(), 2)


def parse_amount(text: str, currency: str) -> int:
    """Parse a decimal string like ``"120.50"`` into minor units.

    Raises ValueError on garbage input or on precision the currency
    cannot represent (e.g. three decimal places for USD).
    """
    try:
        value = Decimal(str(text).strip())
    except (InvalidOperation, ValueError):
        raise ValueError(f"Unparseable amount: {text!r}") from None
    exp = minor_units(currency)
    scaled = value * (10 ** exp)
    if scaled != scaled.to_integral_value():
        raise ValueError(
            f"Amount {text!r} has more precision than {currency.upper()} "
            f"supports ({exp} decimal places).")
    return int(scaled)


def format_amount(minor: int, currency: str) -> str:
    """Exact decimal string for a minor-unit amount (no float rounding)."""
    currency = currency.upper()
    exp = minor_units(currency)
    sign = "-" if minor < 0 else ""
    minor = abs(minor)
    if exp == 0:
        return f"{sign}{minor}"
    whole, frac = divmod(minor, 10 ** exp)
    return f"{sign}{whole}.{str(frac).rjust(exp, '0')}"


def format_money(minor: int, currency: str) -> str:
    """Human form: ``USD 120.50``."""
    return f"{currency.upper()} {format_amount(minor, currency)}"
