"""MarketView: the lookahead barrier, and reading real cached data offline."""

from __future__ import annotations

import datetime as dt
import os

import pytest

from tests.conftest import cached_window
from vela.data.market import FutureData, MarketView

# A read the store already covers costs no vendor requests and never
# authenticates -- antlia enforces that. Setting this makes the guarantee
# explicit, so a cold cache fails the test instead of quietly spending money.
os.environ.setdefault("ANTLIA_HISTORY_FETCH", "0")

# Which symbol and window is local configuration -- see `cached_window`.
# Choose one whose end date clears any expiration that had not expired when
# the cache was warmed; a fill stops on those, so the scopes past them were
# never fetched.


@pytest.fixture(scope="module")
def aapl() -> MarketView:
    symbol, start, end, _ = cached_window()
    try:
        return MarketView.load(symbol, start, end, options=False)
    except Exception as exc:  # noqa: BLE001 -- antlia raises NotCovered by name
        pytest.skip(f"store not warmed for {symbol} {start}..{end}: {exc}")


@pytest.fixture(scope="module")
def aapl_chains() -> MarketView:
    symbol, start, end, max_dte = cached_window()
    try:
        return MarketView.load(symbol, start, end, max_dte=max_dte)
    except Exception as exc:  # noqa: BLE001 -- antlia raises NotCovered by name
        pytest.skip(f"option store not warmed for {symbol} {start}..{end}: {exc}")


def test_sessions_are_trading_days(aapl: MarketView) -> None:
    assert len(aapl.sessions) > 100
    assert aapl.sessions == sorted(aapl.sessions)
    assert all(d.weekday() < 5 for d in aapl.sessions)


def test_pinned_market_refuses_the_future(aapl: MarketView) -> None:
    day = aapl.sessions[100]
    pinned = aapl.pin(day)
    assert pinned.spot().date == day
    assert pinned.bar(aapl.sessions[99]).date == aapl.sessions[99]
    with pytest.raises(FutureData):
        pinned.bar(aapl.sessions[101])


def test_history_ends_at_the_pinned_day(aapl: MarketView) -> None:
    pinned = aapl.pin(aapl.sessions[100])
    past = pinned.history(5)
    assert [b.date for b in past] == aapl.sessions[96:101]


def test_missing_session_is_a_lookup_error(aapl: MarketView) -> None:
    absent = aapl.sessions[-1] + dt.timedelta(days=3650)
    with pytest.raises(LookupError):
        aapl.bar(absent)


def test_chain_is_empty_not_missing_when_unstored(aapl: MarketView) -> None:
    """No contracts today is something to mark against, not a run-ending error."""
    snap = aapl.chain(aapl.sessions[0])
    assert len(snap) == 0
    assert snap.symbol == aapl.symbol


def test_every_session_has_a_chain(aapl_chains: MarketView) -> None:
    assert aapl_chains.sessions_without_chain() == []


def test_a_real_chain_reads_offline(aapl_chains: MarketView) -> None:
    day = aapl_chains.sessions[120]
    snap = aapl_chains.chain(day)
    assert len(snap) > 500
    assert snap.date == day
    assert set(snap.right.tolist()) == {"C", "P"}
    assert snap.dte.min() >= 0


def test_screening_a_real_chain(aapl_chains: MarketView) -> None:
    day = aapl_chains.sessions[120]
    snap, spot = aapl_chains.chain(day), aapl_chains.bar(day)
    otm_puts = snap.select(
        right="P", dte=(30, 45), strikes=(spot.close * 0.90, spot.close * 0.99), quoted=True
    )
    assert len(otm_puts) > 0
    assert (otm_puts.strike < spot.close).all()
    assert otm_puts.mid.min() > 0


def test_locate_round_trips_on_real_data(aapl_chains: MarketView) -> None:
    """A contract named from a row must find that row again."""
    snap = aapl_chains.chain(aapl_chains.sessions[120])
    for i in (0, len(snap) // 2, len(snap) - 1):
        assert snap.locate(snap.contract_at(i)) == i
