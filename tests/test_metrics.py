"""Metrics, and the uncertainty they are supposed to carry."""

from __future__ import annotations

import datetime as dt
import math

import pytest

from vela.book.position import Leg
from vela.engine.results import EquityPoint, Trade
from vela.instrument import Contract
from vela.metrics.curve import curve_stats
from vela.metrics.trades import trade_stats, wilson

EXPIRY = dt.date(2026, 2, 20)
LEG = Leg(Contract("TEST", EXPIRY, 95.0, "P"), -1, 2.0)


def trade(pnl: float, *, margin: float = 500.0, days: int = 10) -> Trade:
    opened = dt.date(2026, 1, 5)
    return Trade(
        position_id=1,
        tag="t",
        legs=(LEG,),
        units=1,
        opened_on=opened,
        closed_on=opened + dt.timedelta(days=days),
        expiration=EXPIRY,
        entry_cash=100.0,
        exit_cash=pnl - 100.0,
        entry_margin=margin,
        reason="test",
    )


def test_wilson_is_not_the_normal_interval() -> None:
    """32 wins of 42 -- the normal approximation runs past 1.0 here."""
    lo, hi = wilson(32, 42)
    assert lo == pytest.approx(0.6147, abs=1e-3)
    assert hi == pytest.approx(0.8652, abs=1e-3)
    assert hi < 1.0


def test_wilson_never_leaves_the_unit_interval() -> None:
    assert wilson(5, 5) == pytest.approx((0.5658, 1.0), abs=1e-3)
    assert wilson(0, 5)[0] == 0.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_trade_stats_reports_uncertainty_not_just_a_mean() -> None:
    trades = [trade(100.0), trade(100.0), trade(100.0), trade(-300.0)]
    stats = trade_stats(trades)
    assert stats.n == 4
    assert stats.wins == 3
    assert stats.win_rate == 0.75
    assert stats.expectancy == pytest.approx(0.0)
    assert stats.expectancy_se > 0
    assert stats.t_stat == pytest.approx(0.0)
    assert not stats.significant
    assert stats.profit_factor == pytest.approx(1.0)
    assert stats.best == 100.0
    assert stats.worst == -300.0


def test_a_high_win_rate_with_no_edge_reads_as_no_edge() -> None:
    """The failure mode this whole module exists to prevent."""
    stats = trade_stats([trade(50.0)] * 9 + [trade(-450.0)])
    assert stats.win_rate == 0.9
    assert stats.total_pnl == pytest.approx(0.0)
    assert not stats.significant


def test_return_on_margin_ignores_trades_with_no_margin() -> None:
    stats = trade_stats([trade(100.0, margin=500.0), trade(100.0, margin=0.0)])
    assert stats.mean_return_on_risk == pytest.approx(0.2)


def test_empty_trades_are_nan_not_zero() -> None:
    """Zero would read as "broke even" -- there was no trade to break even on."""
    stats = trade_stats([])
    assert stats.n == 0
    assert math.isnan(stats.win_rate)
    assert math.isnan(stats.expectancy)
    assert stats.total_pnl == 0.0


def points(values: list[float]) -> list[EquityPoint]:
    start = dt.date(2026, 1, 5)
    return [
        EquityPoint(
            date=start + dt.timedelta(days=i),
            spot=100.0,
            cash=v,
            market_value=0.0,
            realised=0.0,
            unrealised=0.0,
            open_positions=0,
        )
        for i, v in enumerate(values)
    ]


def test_curve_stats_measures_from_the_opening_account() -> None:
    """The first point is recorded after day one traded, so it is not the open."""
    curve = points([99.0, 110.0])
    assert curve_stats(curve, initial=100.0).total_return == pytest.approx(0.10)
    assert curve_stats(curve).total_return == pytest.approx(110 / 99 - 1)


def test_max_drawdown_is_peak_to_trough() -> None:
    stats = curve_stats(points([100.0, 120.0, 90.0, 130.0]), initial=100.0)
    assert stats.max_drawdown == pytest.approx(-0.25)
    assert stats.max_drawdown_sessions == 1


def test_a_flat_curve_has_no_sharpe_rather_than_a_huge_one() -> None:
    stats = curve_stats(points([100.0] * 10), initial=100.0)
    assert stats.total_return == 0.0
    assert math.isnan(stats.sharpe)


def test_a_curve_needs_two_points() -> None:
    with pytest.raises(ValueError, match="two points"):
        curve_stats(points([100.0]))


def test_cash_alone_has_no_excess_and_no_sharpe() -> None:
    """The benchmark accrues exactly as the engine credits: an account that
    never trades matches it to rounding, and rounding is not a Sharpe ratio."""
    from tests.conftest import Scripted, market
    from vela.engine.loop import BacktestEngine

    path = {dt.date(2026, 1, 5) + dt.timedelta(days=i): 100.0 for i in range(0, 30, 3)}
    result = BacktestEngine(cash_rate=0.05).run(market(path), Scripted(lambda ctx: []))
    stats = curve_stats(result.equity, initial=result.initial_cash, risk_free=0.05)
    assert stats.total_return > 0
    assert stats.risk_free_return == pytest.approx(stats.total_return, abs=1e-15)
    assert stats.excess_return == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(stats.sharpe)


def test_ratios_are_measured_against_cash() -> None:
    """An account that only earns the coupon has a positive raw return and no
    edge: against zero it would post a Sharpe made of the coupon alone."""
    curve = points([100.0, 100.05, 100.15, 100.10, 100.25])
    against_zero = curve_stats(curve, initial=100.0)
    against_cash = curve_stats(curve, initial=100.0, risk_free=0.05)
    assert against_cash.total_return == against_zero.total_return
    assert against_cash.volatility == against_zero.volatility
    assert against_cash.risk_free_return == pytest.approx((1 + 0.05 / 365) ** 4 - 1)
    assert against_cash.excess_return == pytest.approx(
        against_cash.total_return - against_cash.risk_free_return
    )
    assert against_cash.sharpe < against_zero.sharpe
