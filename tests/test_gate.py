"""Entry gates, and the lookahead the IV one has to avoid."""

from __future__ import annotations

import datetime as dt
import math

import pytest

from tests.conftest import chain, market
from vela.book.book import BookView
from vela.data.market import MarketView
from vela.engine.context import Context
from vela.engine.loop import BacktestEngine
from vela.execution.fill import MID
from vela.pricing.black_scholes import BlackScholes
from vela.pricing.enrich import YEAR_DAYS, Enricher
from vela.strategy.gate import AlwaysOpen, IVRank
from vela.strategy.vertical import VerticalSpread

BS = BlackScholes()
EXPIRY = dt.date(2026, 3, 20)
START = dt.date(2026, 1, 5)
ENRICH = Enricher(rate=0.04, dividend_yield=0.0)


def priced(day: dt.date, spot: float, vol: float):
    """A chain quoted at Black-Scholes value for `vol`, so IV solves back to it."""
    years = (EXPIRY - day).days / YEAR_DAYS
    rows = []
    for k in (90.0, 95.0, 100.0, 105.0, 110.0):
        for right in ("P", "C"):
            mid = BS.price(spot, k, years, 0.04, vol, right, 0.0)
            rows.append((k, right, mid - 0.02, mid + 0.02))
    return chain(day, {EXPIRY: rows})


def vol_market(vols: list[float], spot: float = 100.0) -> MarketView:
    days = [START + dt.timedelta(days=i) for i in range(len(vols))]
    return market(
        dict.fromkeys(days, spot),
        {d: priced(d, spot, v) for d, v in zip(days, vols, strict=True)},
        enrich=ENRICH,
    )


def context_on(mv: MarketView, index: int) -> Context:
    day = mv.sessions[index]
    return Context.observed(day, mv.pin(day), BookView(cash=100_000.0, positions=()))


def test_always_open_is_always_open() -> None:
    mv = vol_market([0.2, 0.2, 0.2])
    assert AlwaysOpen().is_open(context_on(mv, 2))


def test_the_gate_reads_yesterday_never_today() -> None:
    """The test that matters.

    IV is solved from the 17:15 post-close quotes, so today's is a
    post-close quantity. Here sessions 1-2 are flat and session 3 spikes: a
    gate that only ever sees yesterday finds a flat range on session 3 and
    stays shut. One that peeks at today finds a range and opens.
    """
    mv = vol_market([0.20, 0.20, 0.60])
    gate = IVRank(minimum=0.5, warmup=2)
    for i in range(3):
        gate.observe(context_on(mv, i))

    assert gate.observations == 2  # sessions 1 and 2, not 3
    assert math.isnan(gate.rank())  # a flat range has no rank
    assert not gate.is_open(context_on(mv, 2))


def test_the_spike_is_visible_the_session_after() -> None:
    mv = vol_market([0.20, 0.20, 0.60, 0.60])
    gate = IVRank(minimum=0.5, warmup=2)
    for i in range(4):
        gate.observe(context_on(mv, i))
    assert gate.observations == 3
    assert gate.rank() == pytest.approx(1.0)
    assert gate.is_open(context_on(mv, 3))


def test_the_gate_stays_shut_through_warmup() -> None:
    """A range computed from three observations is not a range."""
    mv = vol_market([0.2, 0.3, 0.4, 0.5, 0.6])
    gate = IVRank(minimum=0.1, warmup=4)
    for i in range(3):
        gate.observe(context_on(mv, i))
    assert math.isnan(gate.rank())
    assert not gate.is_open(context_on(mv, 2))


def test_rank_places_the_latest_value_in_its_own_range() -> None:
    mv = vol_market([0.20, 0.40, 0.30, 0.25])
    gate = IVRank(minimum=0.0, warmup=2)
    for i in range(4):
        gate.observe(context_on(mv, i))
    # observed 0.20, 0.40, 0.30 -> latest 0.30 sits half way up
    assert gate.rank() == pytest.approx(0.5, abs=0.02)


def test_a_shorter_lookback_forgets() -> None:
    mv = vol_market([0.60, 0.20, 0.22, 0.24, 0.26])
    long, short = IVRank(0.0, warmup=2), IVRank(0.0, warmup=2, lookback=3)
    for i in range(5):
        long.observe(context_on(mv, i))
        short.observe(context_on(mv, i))
    # the 0.60 still caps the long window's range, so 0.24 ranks low there
    assert long.rank() < 0.2
    # dropped from the short one, where 0.24 is now the top of its range
    assert short.rank() == pytest.approx(1.0)


def test_a_bad_minimum_is_refused() -> None:
    with pytest.raises(ValueError, match="rank in 0..1"):
        IVRank(minimum=50.0)


# -- through a strategy --------------------------------------------------------


def test_history_advances_on_days_the_gate_is_shut() -> None:
    """Otherwise the range is computed only from days it happened to be open."""
    mv = vol_market([0.2] * 6 + [0.6] * 4)
    gate = IVRank(minimum=0.99, warmup=3)
    strategy = VerticalSpread(
        right="P",
        moneyness=0.95,
        width=5.0,
        entry_dte=(30, 90),
        take_profit=0.5,
        stop_loss=0.5,
        gate=gate,
        tag="g",
    )
    BacktestEngine(fill=MID).run(mv, strategy)
    assert gate.observations == len(mv.sessions) - 1


def test_a_shut_gate_blocks_every_entry() -> None:
    mv = vol_market([0.2] * 8)
    shut = IVRank(minimum=0.99, warmup=3)

    def build(gate):
        return VerticalSpread(
            right="P",
            moneyness=0.95,
            width=5.0,
            entry_dte=(30, 90),
            take_profit=9.0,
            stop_loss=9.0,
            gate=gate,
            max_concurrent=5,
            tag="g",
        )

    assert BacktestEngine(fill=MID).run(mv, build(shut)).equity[-1].open_positions == 0
    opened = BacktestEngine(fill=MID).run(mv, build(AlwaysOpen())).equity[-1].open_positions
    assert opened > 0


def test_the_gate_appears_in_the_strategys_name() -> None:
    """Provenance: two runs that differ only by gate must not read the same."""
    plain = VerticalSpread(
        right="P", moneyness=0.95, width=5.0, entry_dte=(30, 45), take_profit=0.5, stop_loss=0.5
    )
    gated = VerticalSpread(
        right="P",
        moneyness=0.95,
        width=5.0,
        entry_dte=(30, 45),
        take_profit=0.5,
        stop_loss=0.5,
        gate=IVRank(0.5),
    )
    assert plain.name != gated.name
    assert "IVRank" in gated.name


def test_reusing_a_gate_across_runs_does_not_freeze_it() -> None:
    """A stateful gate reused for a second run used to pin itself open.

    Deduplication by date meant every session of the second run was already
    "seen", so the history never advanced and the rank stayed frozen at the
    last value of the first run. The gate then answered the same thing every
    day. Found by reusing one across an envelope pair and getting a gated run
    with every trade in it.
    """
    mv = vol_market([0.20, 0.20, 0.60, 0.60])
    gate = IVRank(minimum=0.5, warmup=2)

    for _ in range(2):
        for i in range(4):
            gate.observe(context_on(mv, i))
        assert gate.observations == 3
        assert gate.rank() == pytest.approx(1.0)


def test_a_stress_pass_does_not_disturb_the_gate() -> None:
    """The intraday pass re-asks on the same day; that is not a new run.

    Treating it as one clears the history on every stressed session, which
    silently gutted a gated envelope run from 22 trades to 6.
    """
    mv = vol_market([0.20, 0.20, 0.60, 0.60])
    gate = IVRank(minimum=0.5, warmup=2)
    for i in range(4):
        gate.observe(context_on(mv, i))
        gate.observe(context_on(mv, i))  # the stress re-ask, same day
    assert gate.observations == 3
    assert gate.rank() == pytest.approx(1.0)
