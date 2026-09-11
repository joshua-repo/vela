"""V1 acceptance: the four properties that stand in for a reference run.

There is no prototype to check numbers against, so correctness rests on
properties instead of a remembered figure:

1. hand-computed reconciliation -- the only test that catches a sign error
2. the cash invariant, asserted every session
3. the expiration payoff identity, against an independently written payoff
4. reproducibility
"""

from __future__ import annotations

import datetime as dt
import random

import pytest

from tests.conftest import Scripted, chain, market
from vela.engine.loop import BacktestEngine
from vela.execution.fill import FillModel
from vela.instrument import Contract
from vela.orders import Close, LegOrder, Open

EXPIRY = dt.date(2026, 2, 20)
SHORT = Contract("TEST", EXPIRY, 95.0, "P")
LONG = Contract("TEST", EXPIRY, 90.0, "P")

FREE = FillModel(haircut=0.0, commission=0.0)
TOUCH = FillModel(haircut=1.0, commission=0.65)


def quotes(spot: float) -> list[tuple[float, str, float, float]]:
    """95P at 2.00/2.40 and 90P at 1.00/1.40 when far from the strikes;
    intrinsic-plus-the-same-spread when they bite."""
    return [
        (95.0, "P", max(95 - spot, 0) + 2.00, max(95 - spot, 0) + 2.40),
        (90.0, "P", max(90 - spot, 0) + 1.00, max(90 - spot, 0) + 1.40),
    ]


def spread_market(path: dict[dt.date, float]):
    return market(path, {d: chain(d, {EXPIRY: quotes(s)}) for d, s in path.items()})


def enter_once(ctx):
    """Open the spread on the first session and never again."""
    if ctx.book.positions or ctx.day != min(PATH):
        return []
    return [Open(legs=(LegOrder(SHORT, -1), LegOrder(LONG, +1)), tag="bps")]


PATH = {
    dt.date(2026, 1, 5): 100.0,
    dt.date(2026, 1, 6): 100.0,
    EXPIRY: 100.0,
}


# -- 1. hand-computed reconciliation ------------------------------------------


def test_credit_is_exactly_the_hand_calculation() -> None:
    """Short 95P at mid 2.20, long 90P at mid 1.20 -> $100 credit on 100x."""
    result = BacktestEngine(fill=FREE).run(spread_market(PATH), Scripted(enter_once))
    trade = result.trades[0]
    assert trade.entry_cash == pytest.approx(100.0)


def test_costs_land_exactly_where_they_should() -> None:
    """At the touch: sell the 95P at its 2.00 bid, buy the 90P at its 1.40 ask.

    200 - 140 = 60 of credit, less 2 contracts x $0.65 = 58.70.
    """
    result = BacktestEngine(fill=TOUCH).run(spread_market(PATH), Scripted(enter_once))
    assert result.trades[0].entry_cash == pytest.approx(58.70)


def test_worthless_expiry_costs_no_exercise_fee() -> None:
    """Both legs expire out of the money, so nothing is exercised."""
    result = BacktestEngine(fill=TOUCH).run(spread_market(PATH), Scripted(enter_once))
    trade = result.trades[0]
    assert trade.reason == "expiration"
    assert trade.exit_cash == pytest.approx(0.0)
    assert trade.pnl == pytest.approx(58.70)


def test_in_the_money_expiry_charges_only_the_exercised_leg() -> None:
    """Settle at 92: the 95P is 3 in the money, the 90P is not."""
    path = {**PATH, EXPIRY: 92.0}
    result = BacktestEngine(fill=TOUCH).run(spread_market(path), Scripted(enter_once))
    trade = result.trades[0]
    # -1 x 3.00 x 100 = -300, and one exercised contract at $0.65
    assert trade.exit_cash == pytest.approx(-300.65)
    assert trade.pnl == pytest.approx(58.70 - 300.65)


def test_closing_early_pays_the_realisable_price_not_the_mid() -> None:
    def script(ctx):
        if not ctx.book.positions and ctx.day == min(PATH):
            return [Open(legs=(LegOrder(SHORT, -1), LegOrder(LONG, +1)), tag="bps")]
        if ctx.book.positions and ctx.day == dt.date(2026, 1, 6):
            return [Close(ctx.book.positions[0].id, reason="take_profit")]
        return []

    result = BacktestEngine(fill=TOUCH).run(spread_market(PATH), Scripted(script))
    trade = result.trades[0]
    assert trade.reason == "take_profit"
    # buy the short back at its 2.40 ask, sell the long at its 1.00 bid:
    # -1 x 2.40 x 100 + 1 x 1.00 x 100 = -140, less 2 x $0.65
    assert trade.exit_cash == pytest.approx(-141.30)


# -- 2. the cash invariant ----------------------------------------------------


def test_cash_invariant_holds_every_session() -> None:
    for fill in (FREE, TOUCH):
        result = BacktestEngine(fill=fill).run(spread_market(PATH), Scripted(enter_once))
        assert result.invariant_breaks() == []


def test_cash_invariant_holds_with_concurrent_positions() -> None:
    """The invariant is only interesting once positions share a cash pool."""
    path = {dt.date(2026, 1, 5) + dt.timedelta(days=i): 100.0 for i in range(10)}
    path[EXPIRY] = 97.0

    def script(ctx):
        if len(ctx.book.positions) < 4 and ctx.day < EXPIRY:
            return [Open(legs=(LegOrder(SHORT, -1), LegOrder(LONG, +1)), tag="ladder")]
        return []

    result = BacktestEngine(fill=TOUCH).run(spread_market(path), Scripted(script))
    assert len(result.trades) == 4
    assert result.invariant_breaks() == []
    assert result.final_equity == pytest.approx(result.initial_cash + result.realised)


# -- 3. the expiration payoff identity ----------------------------------------


def payoff(credit: float, settle: float) -> float:
    """A bull put spread's terminal P&L, written independently of the engine."""
    return credit - max(95.0 - settle, 0.0) * 100 + max(90.0 - settle, 0.0) * 100


@pytest.mark.parametrize("seed", range(25))
def test_expiration_pnl_equals_the_analytic_payoff(seed: int) -> None:
    settle = round(random.Random(seed).uniform(70.0, 130.0), 2)
    path = {**PATH, EXPIRY: settle}
    result = BacktestEngine(fill=FREE).run(spread_market(path), Scripted(enter_once))
    trade = result.trades[0]
    assert trade.pnl == pytest.approx(payoff(trade.entry_cash, settle))


def test_settlement_uses_the_expiration_dates_own_spot() -> None:
    """Not the session the engine happened to notice on.

    The expiration falls on a day with no session; the next session gapped
    hard. Settling against that session instead of the last real price before
    expiry silently books a price the contract never saw.
    """
    before, after = dt.date(2026, 2, 19), dt.date(2026, 2, 24)
    path = {dt.date(2026, 1, 5): 100.0, before: 96.0, after: 60.0}
    result = BacktestEngine(fill=FREE).run(spread_market(path), Scripted(enter_once))
    trade = result.trades[0]
    assert trade.closed_on == EXPIRY
    # settled at 96, not at 60: the 95P is 0 in the money, so the spread keeps
    # its whole credit. Settling at 60 would have cost the full 500 width.
    assert trade.pnl == pytest.approx(trade.entry_cash)


# -- 4. reproducibility -------------------------------------------------------


def test_two_runs_of_the_same_inputs_agree_exactly() -> None:
    def run():
        return BacktestEngine(fill=TOUCH).run(spread_market(PATH), Scripted(enter_once))

    a, b = run(), run()
    assert [t.pnl for t in a.trades] == [t.pnl for t in b.trades]
    assert [(p.date, p.equity) for p in a.equity] == [(p.date, p.equity) for p in b.equity]


# -- 5. cost attribution ------------------------------------------------------


def test_costs_are_attributed_not_just_netted() -> None:
    """The gap between gross and net is the whole argument about execution."""
    result = BacktestEngine(fill=TOUCH).run(spread_market(PATH), Scripted(enter_once))
    trade = result.trades[0]
    # entry crosses a full half-spread on both legs: 2 legs x $0.20 x 100
    assert trade.costs.spread == pytest.approx(40.0)
    # 2 contracts in, and nothing exercised at expiry, so nothing out
    assert trade.costs.commission == pytest.approx(1.30)
    assert trade.pnl == pytest.approx(trade.gross_pnl - trade.costs.total)
    assert trade.gross_pnl == pytest.approx(100.0)  # the uncosted credit


def test_a_free_fill_model_attributes_nothing() -> None:
    result = BacktestEngine(fill=FREE).run(spread_market(PATH), Scripted(enter_once))
    assert result.costs.total == 0.0
    assert result.trades[0].gross_pnl == pytest.approx(result.trades[0].pnl)


def test_marking_a_held_position_accrues_no_cost() -> None:
    """A mark is not a fill. Charging for one would bill every held day."""
    path = {dt.date(2026, 1, 5) + dt.timedelta(days=i): 100.0 for i in range(20)}
    path[EXPIRY] = 100.0
    result = BacktestEngine(fill=TOUCH).run(spread_market(path), Scripted(enter_once))
    # one entry, one expiry: the entry's two crossings and nothing more
    assert result.costs.spread == pytest.approx(40.0)


def test_exercise_fees_land_only_on_settlement() -> None:
    path = {**PATH, EXPIRY: 92.0}
    result = BacktestEngine(fill=TOUCH).run(spread_market(path), Scripted(enter_once))
    trade = result.trades[0]
    # 2 contracts opened, plus the one in-the-money leg exercised
    assert trade.costs.commission == pytest.approx(3 * 0.65)
    assert trade.costs.spread == pytest.approx(40.0)
