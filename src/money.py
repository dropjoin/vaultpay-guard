"""Money handling — integer minor units only, never floats.

Every amount inside VaultPay Guard is an ``int`` of minor units (cents
for USD/EUR/GBP, whole yen for JPY, ...) plus an ISO currency code.
Parsing goes through :class:`decimal.Decimal` so a value like ``0.1``
can never pick up binary float error on its way into a payment
decision — the same discipline VaultMind applied to wei amounts,
adapted to fiat.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, DecimalException

# ISO 4217 minor-unit exponents for the currencies the demo touches.
# Unknown currencies default to 2 (the overwhelmingly common case).
MINOR_UNITS: dict[str, int] = {
    "USD": 2, "EUR": 2, "GBP": 2, "SGD": 2, "AUD": 2, "CAD": 2,
    "IDR": 2, "JPY": 0, "KRW": 0,
}

# Largest magnitude parse_amount accepts: 10**15 in major units
# (~a quadrillion dollars) is far above any real payment, cap or
# balance this guard will ever see, and it keeps the int() conversion
# below ~18 digits so a compact exponent ("1e999999") can never turn
# into a gigantic integer allocation.
_MAX_ADJUSTED_EXPONENT = 15


def minor_units(currency: str) -> int:
    return MINOR_UNITS.get(currency.upper(), 2)


def parse_amount(text: str, currency: str) -> int:
    """Parse a decimal string like ``"120.50"`` into minor units.

    Raises ValueError on garbage input, on non-finite values
    (``NaN``/``Infinity``), on magnitudes no payment could ever have,
    or on precision the currency cannot represent (e.g. three decimal
    places for USD). Every failure is a ValueError: callers rely on
    that contract to map bad input to a 422 / ProposalError instead
    of crashing on a stray ``decimal`` exception or OverflowError.
    """
    try:
        value = Decimal(str(text).strip())
    except (InvalidOperation, ValueError):
        raise ValueError(f"Unparseable amount: {text!r}") from None
    if not value.is_finite():
        raise ValueError(f"Unparseable amount: {text!r}")
    if value != 0 and value.adjusted() > _MAX_ADJUSTED_EXPONENT:
        raise ValueError(f"Amount {text!r} exceeds the supported range.")
    exp = minor_units(currency)
    try:
        scaled = value * (10 ** exp)
        integral = scaled.to_integral_value()
    except DecimalException:
        raise ValueError(f"Unparseable amount: {text!r}") from None
    if scaled != integral:
        raise ValueError(
            f"Amount {text!r} has more precision than {currency.upper()} "
            f"supports ({exp} decimal places).")
    try:
        return int(scaled)
    except (OverflowError, ValueError):
        raise ValueError(f"Unparseable amount: {text!r}") from None


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
