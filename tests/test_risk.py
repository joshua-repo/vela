"""Margin and sizing: the two questions a defined-risk structure answers alike."""

from __future__ import annotations

import datetime as dt
import math

import pytest

from vela.book.payoff import terminal_bounds
from vela.book.position import Leg
from vela.instrument import Contract
from vela.risk.margin import MaxLossMargin, NoMargin, UnboundedRisk
from vela.risk.sizing import FixedRisk, FixedUnits

EXPIRY = dt.date(2026, 2, 20)


def put(strike: float, qty: int, price: float) -> Leg:
    return Leg(Contract("TEST", EXPIRY, strike, "P"), qty, price)


def call(strike: float, qty: int, price: float) -> Leg:
    return Leg(Contract("TEST", EXPIRY, strike, "C"), qty, price)


# short the 95 put at 2.20, long the 90 put at 1.20 -> $100 credit, $5 wide
SPREAD = [put(95.0, -1, 2.20), put(90.0, +1, 1.20)]


def test_a_vertical_margin_is_exactly_reg_t() -> None:
    """Width x multiplier, less the credit -- derived from the payoff, not copied."""
    assert MaxLossMargin().requirement(SPREAD, spot=100.0) == pytest.approx(500 - 100)


def test_max_profit_of_a_credit_spread_is_the_credit() -> None:
    """The bound a profit target must be measured against."""
    worst, best = terminal_bounds(SPREAD)
    assert best == pytest.approx(100.0)
    assert worst == pytest.approx(-400.0)


def test_a_debit_spread_bounds_the_other_way() -> None:
    """Long the 100 call at 3.00, short the 105 at 1.00: $200 paid, $300 to make."""
    debit = [call(100.0, +1, 3.00), call(105.0, -1, 1.00)]
    worst, best = terminal_bounds(debit)
    assert worst == pytest.approx(-200.0)
    assert best == pytest.approx(300.0)
    assert MaxLossMargin().requirement(debit, spot=100.0) == pytest.approx(200.0)


def test_a_naked_short_call_is_refused_not_approximated() -> None:
    with pytest.raises(UnboundedRisk, match="no bounded worst case"):
        MaxLossMargin().requirement([call(105.0, -1, 1.00)], spot=100.0)


def test_a_cash_secured_put_is_bounded_and_named_honestly() -> None:
    """Bounded at zero spot -- the cash-secured amount, not Reg-T's naked rule."""
    assert MaxLossMargin().requirement([put(95.0, -1, 2.20)], spot=100.0) == pytest.approx(
        9500 - 220
    )


def test_a_long_option_can_win_without_bound() -> None:
    _, best = terminal_bounds([call(100.0, +1, 3.00)])
    assert best == math.inf


def test_no_margin_ties_up_nothing() -> None:
    assert NoMargin().requirement(SPREAD, spot=100.0) == 0.0


def test_fixed_risk_sizes_to_the_budget() -> None:
    """$400 of risk per unit, $1000 budget -> 2 units."""
    units = FixedRisk(1000.0).units(
        unit_cash=100.0, unit_margin=400.0, available_cash=100_000.0, max_units=None
    )
    assert units == 2


def test_fixed_risk_respects_the_strategys_cap() -> None:
    units = FixedRisk(1000.0).units(
        unit_cash=100.0, unit_margin=400.0, available_cash=100_000.0, max_units=1
    )
    assert units == 1


def test_sizing_declines_rather_than_overdrawing() -> None:
    """0 is a normal answer: an account that cannot afford a structure skips it."""
    units = FixedRisk(1000.0).units(
        unit_cash=100.0, unit_margin=400.0, available_cash=100.0, max_units=None
    )
    assert units == 0


def test_fixed_risk_cannot_size_against_zero_risk() -> None:
    units = FixedRisk(1000.0).units(
        unit_cash=100.0, unit_margin=0.0, available_cash=100_000.0, max_units=None
    )
    assert units == 0


def test_fixed_units_still_checks_affordability() -> None:
    assert (
        FixedUnits(5).units(
            unit_cash=100.0, unit_margin=400.0, available_cash=100_000.0, max_units=None
        )
        == 5
    )
    assert (
        FixedUnits(5).units(
            unit_cash=100.0, unit_margin=400.0, available_cash=900.0, max_units=None
        )
        == 2
    )
