"""The intraday stress pass: the pessimistic half of the envelope."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from tests.conftest import Scripted, bar, chain, market
from vela.book.position import Leg, Position
from vela.data.snapshot import ChainSnapshot
from vela.engine.intraday import eligible, leg_ivs, stress_chain
from vela.engine.loop import BacktestEngine
from vela.execution.fill import MID
from vela.instrument import Contract
from vela.orders import Close
from vela.pricing.black_scholes import BlackScholes
from vela.pricing.enrich import YEAR_DAYS, Enricher
from vela.strategy.vertical import VerticalSpread

BS = BlackScholes()
REASON = "stop_loss"
EXPIRY = dt.date(2026, 3, 20)
DAY = dt.date(2026, 1, 5)
SHORT = Contract("TEST", EXPIRY, 95.0, "P")
LONG = Contract("TEST", EXPIRY, 90.0, "P")
ENRICH = Enricher(rate=0.04, dividend_yield=0.0)


def priced(day: dt.date, spot: float, vol: float = 0.25) -> ChainSnapshot:
    years = (EXPIRY - day).days / YEAR_DAYS
    rows = []
    for k in (85.0, 90.0, 95.0, 100.0, 105.0):
        for right in ("P", "C"):
            mid = BS.price(spot, k, years, 0.04, vol, right, 0.0)
            rows.append((k, right, mid - 0.10, mid + 0.10))
    return chain(day, {EXPIRY: rows})


def spread() -> Position:
    return Position(
        id=1,
        legs=(Leg(SHORT, -1, 2.0), Leg(LONG, +1, 1.0)),
        opened_on=DAY - dt.timedelta(days=1),
        entry_spot=100.0,
        entry_cash=100.0,
    )


def test_stress_reprices_only_the_open_legs() -> None:
    snap = priced(DAY, 100.0)
    before = snap.mid.copy()
    stressed = stress_chain(
        snap,
        [spread()],
        low=90.0,
        day=DAY,
        prior_iv={SHORT: 0.25, LONG: 0.25},
        model=BS,
        rate=0.04,
        dividend_yield=0.0,
    )
    moved = ~np.isclose(before, stressed.mid, equal_nan=True)
    touched = {stressed.contract_at(int(i)) for i in np.flatnonzero(moved)}
    assert touched == {SHORT, LONG}


def test_a_lower_spot_makes_a_short_put_spread_worse() -> None:
    snap = priced(DAY, 100.0)
    position = spread()
    stressed = stress_chain(
        snap,
        [position],
        low=90.0,
        day=DAY,
        prior_iv={SHORT: 0.25, LONG: 0.25},
        model=BS,
        rate=0.04,
        dividend_yield=0.0,
    )
    at_close = position.market_value(snap, 100.0, MID)
    at_low = position.market_value(stressed, 90.0, MID)
    assert at_low < at_close


def test_the_observed_spread_is_preserved() -> None:
    """The level moves; what the market was quoting to cross does not."""
    snap = priced(DAY, 100.0)
    stressed = stress_chain(
        snap,
        [spread()],
        low=90.0,
        day=DAY,
        prior_iv={SHORT: 0.25, LONG: 0.25},
        model=BS,
        rate=0.04,
        dividend_yield=0.0,
    )
    row = snap.locate(SHORT)
    assert row is not None
    assert stressed.ask[row] - stressed.bid[row] == pytest.approx(snap.ask[row] - snap.bid[row])


def test_a_leg_with_no_prior_iv_is_left_alone() -> None:
    """Substituting today's IV would import the lookahead this avoids."""
    snap = priced(DAY, 100.0)
    stressed = stress_chain(
        snap,
        [spread()],
        low=90.0,
        day=DAY,
        prior_iv={SHORT: 0.25},  # nothing for the long leg
        model=BS,
        rate=0.04,
        dividend_yield=0.0,
    )
    row = snap.locate(LONG)
    assert row is not None
    assert stressed.bid[row] == snap.bid[row]


def test_leg_ivs_collects_what_is_solvable() -> None:
    snap = priced(DAY, 100.0, vol=0.30)
    found = leg_ivs(snap, ENRICH.greeks(snap, 100.0), [spread()])
    assert set(found) == {SHORT, LONG}
    assert found[SHORT] == pytest.approx(0.30, abs=5e-3)


# -- through the engine --------------------------------------------------------


def dipping_market(low_on_day_two: float):
    """Flat closes, but session two dips intraday and recovers."""
    days = [DAY, DAY + dt.timedelta(days=1), DAY + dt.timedelta(days=2)]
    bars = {d: bar(d, 100.0) for d in days}
    import dataclasses

    bars[days[1]] = dataclasses.replace(bars[days[1]], low=low_on_day_two)
    view = market({d: 100.0 for d in days}, {d: priced(d, 100.0) for d in days}, enrich=ENRICH)
    return dataclasses.replace(view, _bars=bars)


def strategy(**kw):
    params = dict(
        right="P",
        moneyness=0.95,
        width=5.0,
        entry_dte=(30, 90),
        take_profit=9.0,
        stop_loss=0.15,
        tag="v",
    )
    params.update(kw)
    return VerticalSpread(**params)  # type: ignore[arg-type]


def test_a_dip_the_close_never_saw_produces_an_intraday_stop() -> None:
    view = dipping_market(80.0)
    close_only = BacktestEngine(fill=MID).run(view, strategy())
    intraday = BacktestEngine(fill=MID, intraday_stop=True).run(view, strategy())

    assert [t.reason for t in close_only.trades] == []
    assert [t.reason for t in intraday.trades] == ["stop_loss_intraday"]


def test_no_dip_means_no_difference() -> None:
    view = dipping_market(99.9)
    a = BacktestEngine(fill=MID).run(view, strategy())
    b = BacktestEngine(fill=MID, intraday_stop=True).run(view, strategy())
    assert [t.pnl for t in a.trades] == [t.pnl for t in b.trades]


def test_a_position_opened_today_cannot_stop_at_an_earlier_low() -> None:
    """The low printed hours before the position existed."""
    today = DAY + dt.timedelta(days=1)
    opened_before = Position(
        id=1, legs=(Leg(SHORT, -1, 2.0),), opened_on=DAY, entry_spot=100.0, entry_cash=200.0
    )
    opened_today = Position(
        id=2, legs=(Leg(SHORT, -1, 2.0),), opened_on=today, entry_spot=100.0, entry_cash=200.0
    )
    assert eligible([opened_before, opened_today], today) == [opened_before]
    assert eligible([opened_today], today) == []


def test_only_a_stop_is_acted_on_at_the_low() -> None:
    """A take-profit at the low is a limit order nobody can assume filled.

    Driven by a strategy that answers *only* on the stressed pass, so the
    close pass cannot pre-empt it -- an earlier version of this test set an
    unreachable stop and a trivial take-profit, which simply closed on the
    close pass and never exercised the filter at all.
    """
    view = dipping_market(80.0)

    class OnlyAtTheLow(Scripted):
        pass

    def decide(ctx):
        if not ctx.book.positions and not ctx.stressed:
            return strategy().on_day(ctx)
        if ctx.stressed and ctx.book.positions:
            return [Close(ctx.book.positions[0].id, reason=REASON)]
        return []

    global REASON
    REASON = "take_profit"
    profit_run = BacktestEngine(fill=MID, intraday_stop=True).run(view, Scripted(decide))
    assert all("intraday" not in t.reason for t in profit_run.trades)

    REASON = "stop_loss"
    stop_run = BacktestEngine(fill=MID, intraday_stop=True).run(view, Scripted(decide))
    assert [t.reason for t in stop_run.trades] == ["stop_loss_intraday"]


def test_intraday_needs_solved_volatilities() -> None:
    days = [DAY, DAY + dt.timedelta(days=1)]
    bare = market({d: 100.0 for d in days}, {d: priced(d, 100.0) for d in days})
    with pytest.raises(ValueError, match="enrich"):
        BacktestEngine(fill=MID, intraday_stop=True).run(bare, strategy())
