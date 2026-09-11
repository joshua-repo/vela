"""Early assignment: the short leg taken away before you meant to close it."""

from __future__ import annotations

import datetime as dt

import pytest

from tests.conftest import Scripted, chain, market
from vela.book.position import Leg, Position
from vela.engine.loop import BacktestEngine
from vela.execution.assignment import NoAssignment, ParityAssignment
from vela.execution.fill import MID, FillModel
from vela.instrument import Contract
from vela.orders import Close, LegOrder, Open

EXPIRY = dt.date(2026, 2, 20)
ENTRY = dt.date(2026, 1, 5)
SHORT = Contract("TEST", EXPIRY, 95.0, "P")
LONG = Contract("TEST", EXPIRY, 90.0, "P")


def spread_at(day: dt.date, spot: float, *, extrinsic: float = 2.0):
    """The 95/90 put spread quoted at intrinsic plus `extrinsic`, +/- 0.20."""
    rows = []
    for strike in (95.0, 90.0):
        mid = max(strike - spot, 0.0) + extrinsic
        rows.append((strike, "P", mid - 0.20, mid + 0.20))
    return chain(day, {EXPIRY: rows})


def position_at(spot: float) -> Position:
    return Position(
        id=1,
        legs=(Leg(SHORT, -1, 2.20), Leg(LONG, +1, 1.20)),
        opened_on=ENTRY,
        entry_spot=100.0,
        entry_cash=100.0,
    )


def test_parity_assignment_fires_when_extrinsic_is_gone() -> None:
    snap = spread_at(ENTRY, 88.0, extrinsic=0.0)
    assigned = ParityAssignment().assigned(position_at(88.0), snap, 88.0)
    assert [leg.contract for leg in assigned] == [SHORT]


def test_it_does_not_fire_while_time_value_remains() -> None:
    snap = spread_at(ENTRY, 88.0, extrinsic=2.0)
    assert ParityAssignment().assigned(position_at(88.0), snap, 88.0) == ()


def test_a_threshold_assigns_earlier() -> None:
    snap = spread_at(ENTRY, 88.0, extrinsic=0.5)
    assert ParityAssignment(threshold=0.0).assigned(position_at(88.0), snap, 88.0) == ()
    assert ParityAssignment(threshold=1.0).assigned(position_at(88.0), snap, 88.0) != ()


def test_out_of_the_money_shorts_are_never_assigned() -> None:
    """Nobody exercises a worthless right, whatever the quote looks like."""
    snap = spread_at(ENTRY, 110.0, extrinsic=0.0)
    assert ParityAssignment().assigned(position_at(110.0), snap, 110.0) == ()


def test_long_legs_are_never_assigned() -> None:
    """Assignment happens *to* a writer. The long leg is the holder's choice."""
    long_only = Position(
        id=1, legs=(Leg(LONG, +1, 1.20),), opened_on=ENTRY, entry_spot=100.0, entry_cash=-120.0
    )
    snap = spread_at(ENTRY, 80.0, extrinsic=0.0)
    assert ParityAssignment().assigned(long_only, snap, 80.0) == ()


def test_a_missing_quote_is_not_evidence_of_parity() -> None:
    """Otherwise a data gap terminates positions."""
    snap = chain(ENTRY, {EXPIRY: [(90.0, "P", 1.0, 1.2)]})  # the 95 is absent
    assert ParityAssignment().assigned(position_at(88.0), snap, 88.0) == ()


def test_no_assignment_model_never_fires() -> None:
    snap = spread_at(ENTRY, 88.0, extrinsic=0.0)
    assert NoAssignment().assigned(position_at(88.0), snap, 88.0) == ()


# -- through the engine -------------------------------------------------------

PATH = {ENTRY: 100.0, dt.date(2026, 1, 6): 88.0, EXPIRY: 88.0}


def market_with(extrinsic_on_day_two: float):
    return market(
        PATH,
        {
            ENTRY: spread_at(ENTRY, 100.0, extrinsic=2.0),
            dt.date(2026, 1, 6): spread_at(
                dt.date(2026, 1, 6), 88.0, extrinsic=extrinsic_on_day_two
            ),
            EXPIRY: spread_at(EXPIRY, 88.0, extrinsic=0.0),
        },
    )


def enter_once(ctx):
    if ctx.book.positions or ctx.day != ENTRY:
        return []
    return [Open(legs=(LegOrder(SHORT, -1), LegOrder(LONG, +1)), tag="bps")]


def test_the_engine_terminates_an_assigned_position() -> None:
    result = BacktestEngine(fill=MID).run(market_with(0.0), Scripted(enter_once))
    trade = result.trades[0]
    assert trade.reason == "early_assignment"
    assert trade.closed_on == dt.date(2026, 1, 6)


def test_no_assignment_lets_the_position_ride_to_expiry() -> None:
    result = BacktestEngine(fill=MID, assignment=NoAssignment()).run(
        market_with(0.0), Scripted(enter_once)
    )
    assert result.trades[0].reason == "expiration"


def test_a_strategy_that_closes_first_is_never_assigned() -> None:
    """You are assigned overnight; a position closed during the day is gone."""

    def script(ctx):
        if not ctx.book.positions and ctx.day == ENTRY:
            return enter_once(ctx)
        if ctx.book.positions and ctx.day == dt.date(2026, 1, 6):
            return [Close(ctx.book.positions[0].id, reason="stop_loss")]
        return []

    result = BacktestEngine(fill=MID).run(market_with(0.0), Scripted(script))
    assert result.trades[0].reason == "stop_loss"


def test_assignment_settles_shorts_at_intrinsic_and_sells_the_long() -> None:
    """No spread is crossed on the assigned leg -- the holder just exercises."""
    result = BacktestEngine(fill=FillModel(haircut=1.0, commission=0.0)).run(
        market_with(0.0), Scripted(enter_once)
    )
    trade = result.trades[0]
    # at spot 88 both legs are in the money. The short 95P is assigned at its
    # intrinsic 7.00 -> -700, crossing nothing. The long 90P (intrinsic 2.00)
    # is sold into its 1.80 bid -> +180.
    assert trade.exit_cash == pytest.approx(-700.0 + 180.0)
    # $20 on the long sale, on top of the $40 crossed opening both legs
    assert trade.costs.spread == pytest.approx(20.0 + 40.0)
