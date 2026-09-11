"""VerticalSpread: one structure, two rights, and exits on the right scale."""

from __future__ import annotations

import datetime as dt
import math

import pytest

from tests.conftest import Scripted, chain, market
from vela.engine.loop import BacktestEngine
from vela.execution.fill import MID, FillModel
from vela.risk.margin import MaxLossMargin
from vela.risk.sizing import FixedUnits
from vela.strategy.vertical import VerticalSpread

EXPIRY = dt.date(2026, 2, 20)
ENTRY = dt.date(2026, 1, 5)


def full_chain(day: dt.date, spot: float):
    """Puts and calls every $5 from 80 to 120: intrinsic plus decaying extrinsic.

    Extrinsic decays exponentially with distance from spot, which is the one
    property a synthetic chain has to get right here: a flat time value makes
    every vertical worth exactly zero, and a Gaussian one makes a bull put
    spread gain value as spot rises. Both would make these tests assert
    nonsense about a market that cannot exist.
    """
    rows = []
    for k in range(80, 125, 5):
        strike = float(k)
        extrinsic = 4.0 * math.exp(-abs(strike - spot) / 10.0)
        for right, intrinsic in (("P", max(strike - spot, 0.0)), ("C", max(spot - strike, 0.0))):
            mid = intrinsic + extrinsic
            rows.append((strike, right, mid - 0.20, mid + 0.20))
    return chain(day, {EXPIRY: rows})


def one_entry_market(path: dict[dt.date, float]):
    return market(path, {d: full_chain(d, s) for d, s in path.items()})


def strategy(**kw):
    params = dict(
        right="P",
        moneyness=0.95,
        width=5.0,
        entry_dte=(30, 60),
        take_profit=0.5,
        stop_loss=0.5,
        tag="v",
    )
    params.update(kw)
    return VerticalSpread(**params)  # type: ignore[arg-type]


def run(strat, path, **engine_kw):
    engine = BacktestEngine(fill=MID, margin=MaxLossMargin(), sizing=FixedUnits(1), **engine_kw)
    return engine.run(one_entry_market(path), strat)


def test_a_bull_put_sells_below_spot_and_buys_lower() -> None:
    result = run(strategy(), {ENTRY: 100.0, EXPIRY: 100.0})
    legs = result.trades[0].legs
    short = next(leg for leg in legs if leg.quantity < 0)
    long = next(leg for leg in legs if leg.quantity > 0)
    assert short.contract.right == "P"
    assert short.contract.strike == 95.0  # nearest to 0.95 x 100
    assert long.contract.strike == 90.0  # width below
    assert result.trades[0].entry_cash > 0  # a credit


def test_a_bull_call_is_the_same_shape_at_a_different_moneyness() -> None:
    """Short the higher strike, long the one below -- a debit, and bullish."""
    result = run(strategy(right="C", moneyness=1.05), {ENTRY: 100.0, EXPIRY: 100.0})
    legs = result.trades[0].legs
    short = next(leg for leg in legs if leg.quantity < 0)
    long = next(leg for leg in legs if leg.quantity > 0)
    assert short.contract.right == "C"
    assert short.contract.strike == 105.0
    assert long.contract.strike == 100.0
    assert result.trades[0].entry_cash < 0  # a debit


def test_take_profit_is_a_fraction_of_max_profit_not_of_risk() -> None:
    """The bug the first real-data run exposed: a target that can never fire.

    The spread's credit is $100 against $400 of risk. A target of half the max
    profit is $50 and is reachable; half the risk would be $200 and is not.
    """
    path = {ENTRY: 100.0, dt.date(2026, 1, 6): 110.0, EXPIRY: 110.0}
    reachable = run(strategy(take_profit=0.5), path)
    assert reachable.trades[0].reason == "take_profit"
    assert reachable.trades[0].closed_on == dt.date(2026, 1, 6)

    # 1.5x max profit is unreachable by construction, so it rides to expiry
    unreachable = run(strategy(take_profit=1.5), path)
    assert unreachable.trades[0].reason == "expiration"


def test_stop_loss_fires_on_a_fraction_of_max_loss() -> None:
    path = {ENTRY: 100.0, dt.date(2026, 1, 6): 88.0, EXPIRY: 88.0}
    result = run(strategy(take_profit=9.0, stop_loss=0.5), path)
    assert result.trades[0].reason == "stop_loss"
    assert result.trades[0].closed_on == dt.date(2026, 1, 6)


def test_time_exit_closes_before_expiry() -> None:
    path = {ENTRY: 100.0, dt.date(2026, 2, 16): 100.0, EXPIRY: 100.0}
    result = run(strategy(take_profit=9.0, exit_dte=7), path)
    assert result.trades[0].reason == "time_exit"
    assert result.trades[0].closed_on == dt.date(2026, 2, 16)


def test_max_concurrent_caps_the_book() -> None:
    path = {ENTRY + dt.timedelta(days=i): 100.0 for i in range(10)}
    path[EXPIRY] = 100.0
    result = run(strategy(take_profit=9.0, max_concurrent=3), path)
    assert max(point.open_positions for point in result.equity) == 3


def test_entries_respect_the_spacing_rule() -> None:
    path = {ENTRY + dt.timedelta(days=i): 100.0 for i in range(12)}
    path[EXPIRY] = 100.0
    result = run(strategy(take_profit=9.0, max_concurrent=10, min_days_between_entries=5), path)
    opened = sorted({t.opened_on for t in result.trades})
    gaps = [(b - a).days for a, b in zip(opened, opened[1:], strict=False)]
    assert all(gap >= 5 for gap in gaps)


def test_no_entry_when_the_long_strike_is_missing() -> None:
    """Declining is correct: half a spread is a different structure."""
    thin = market(
        {ENTRY: 100.0, EXPIRY: 100.0},
        {d: chain(d, {EXPIRY: [(95.0, "P", 2.20, 2.60)]}) for d in (ENTRY, EXPIRY)},
    )
    result = BacktestEngine(fill=MID, margin=MaxLossMargin()).run(thin, strategy())
    assert result.trades == []


def test_positions_are_marked_at_the_mid_whatever_the_fill_model() -> None:
    """The strategy's view of a position does not move with the haircut.

    Marking at the side the engine would actually fill at would make a
    strategy hold every winner half a spread longer, for no reason a market
    would recognise.
    """
    seen: dict[str, float] = {}

    def watch(ctx):
        if not ctx.book.positions:
            return strategy().on_day(ctx)
        seen[str(ctx.day)] = ctx.unrealised_at_mid(ctx.book.positions[0])
        return []

    path = {ENTRY: 100.0, dt.date(2026, 1, 6): 110.0, EXPIRY: 110.0}
    values = []
    for haircut in (0.0, 1.0):
        seen.clear()
        strat = Scripted(watch)
        strat.name = "watch"
        BacktestEngine(fill=FillModel(haircut=haircut, commission=0.0), margin=MaxLossMargin()).run(
            one_entry_market(path), strat
        )
        values.append(seen["2026-01-06"])

    # Entry prices differ with the haircut, so the P&L does -- but the *mark*
    # on the far side is the mid in both, so the gap is exactly the entry
    # haircut and nothing more.
    assert values[0] != values[1]
    assert values[0] - values[1] == pytest.approx(2 * 0.20 * 100)


def test_costs_reduce_realised_pnl_without_changing_the_rule() -> None:
    path = {ENTRY: 100.0, dt.date(2026, 1, 6): 110.0, EXPIRY: 110.0}
    costed = BacktestEngine(
        fill=FillModel(haircut=0.25, commission=0.65), margin=MaxLossMargin()
    ).run(one_entry_market(path), strategy(take_profit=0.5))
    free = run(strategy(take_profit=0.5), path)
    assert costed.trades[0].reason == free.trades[0].reason == "take_profit"
    assert costed.trades[0].pnl < free.trades[0].pnl
