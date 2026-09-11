"""Data provenance: which append of the store a run read."""

from __future__ import annotations

import datetime as dt
import os

import pytest

from tests.conftest import cached_window
from vela.data.market import DataVersion, MarketView
from vela.run.cli import _as_of

os.environ.setdefault("ANTLIA_HISTORY_FETCH", "0")

#: Which symbol and window is local configuration -- see `cached_window`.
STAMP = dt.datetime(2026, 9, 2, 19, 20, tzinfo=dt.UTC)


def test_an_unpinned_verified_read_says_so() -> None:
    version = DataVersion(tables={"equity_eod": STAMP}, coverage_verified=True)
    assert not version.pinned
    assert "coverage unverified" not in str(version)


def test_an_unverified_read_says_so() -> None:
    assert "coverage unverified" in str(DataVersion(tables={"equity_eod": STAMP}))


def test_a_pinned_read_never_claims_verified_coverage() -> None:
    """Coverage comes from the ledger, which knows nothing about `as_of`.

    The ledger records every scope that has ever settled, so a pinned read
    sees a subset of what coverage is describing. Measured on this store: at
    the midpoint of its own warm-up, coverage reported complete while the read
    returned 198,468 of 375,486 rows.
    """
    version = DataVersion(tables={"option_eod": STAMP}, as_of=STAMP, coverage_verified=False)
    assert version.pinned
    assert "coverage unverifiable" in str(version)


def test_an_unknown_version_is_reported_not_hidden() -> None:
    assert "unknown" in str(DataVersion(tables={"equity_eod": None}))


def test_as_of_reads_a_naive_timestamp_as_utc() -> None:
    """antlia stamps every append in UTC; a naive pin would be silently off."""
    assert _as_of("2026-09-02T18:25:15") == dt.datetime(2026, 9, 2, 18, 25, 15, tzinfo=dt.UTC)
    assert _as_of("2026-09-02T18:25:15Z") == dt.datetime(2026, 9, 2, 18, 25, 15, tzinfo=dt.UTC)
    assert _as_of(None) is None


def test_a_pinned_offset_is_respected() -> None:
    got = _as_of("2026-09-02T14:25:15-04:00")
    assert got == dt.datetime(2026, 9, 2, 18, 25, 15, tzinfo=dt.UTC)


# -- against the real store ----------------------------------------------------


@pytest.fixture(scope="module")
def loaded() -> MarketView:
    symbol, start, end, max_dte = cached_window()
    try:
        return MarketView.load(symbol, start, end, max_dte=max_dte)
    except Exception as exc:  # noqa: BLE001 -- antlia raises NotCovered by name
        pytest.skip(f"store not warmed for {symbol} {start}..{end}: {exc}")


def test_a_real_load_records_a_version_for_every_table(loaded: MarketView) -> None:
    assert set(loaded.version.tables) == {"equity_eod", "option_eod"}
    assert all(v is not None for v in loaded.version.tables.values())
    assert loaded.version.coverage_verified
    assert not loaded.version.pinned


def test_pinning_to_the_newest_append_changes_nothing(loaded: MarketView) -> None:
    newest = loaded.version.tables["option_eod"]
    assert newest is not None
    symbol, start, end, max_dte = cached_window()
    pinned = MarketView.load(symbol, start, end, max_dte=max_dte, as_of=newest)
    assert [len(pinned.chain(d)) for d in pinned.sessions] == [
        len(loaded.chain(d)) for d in loaded.sessions
    ]


def test_pinning_earlier_returns_less_and_records_what_it_read(loaded: MarketView) -> None:
    """The whole point: a run can be replayed against what the store said then.

    Asserts the recorded stamp too, not just the row count -- a version that
    always reports the newest append would make every pinned run claim to have
    read data it did not.
    """
    symbol, start, end, max_dte = cached_window()
    pinned = MarketView.load(symbol, start, end, max_dte=max_dte, as_of=_midpoint(), verify=False)

    assert sum(len(pinned.chain(d)) for d in pinned.sessions) < sum(
        len(loaded.chain(d)) for d in loaded.sessions
    )
    midpoint = _midpoint()
    recorded = pinned.version.tables["option_eod"]
    newest = loaded.version.tables["option_eod"]
    assert recorded is not None and newest is not None
    assert recorded <= midpoint
    assert recorded < newest
    assert pinned.version.pinned


def test_a_pinned_read_is_unverified_even_when_verification_runs(loaded: MarketView) -> None:
    """`verify=True` still runs -- it is an upper bound -- but it cannot vouch.

    Coverage is computed from the ledger, which records every scope that has
    ever settled and knows nothing about `as_of`. Here it passes (the live
    store is complete) while the pinned read sees roughly half the rows, so a
    result that took the pass at face value would be claiming verification it
    never had.
    """
    symbol, start, end, max_dte = cached_window()
    pinned = MarketView.load(symbol, start, end, max_dte=max_dte, as_of=_midpoint(), verify=True)
    assert not pinned.version.coverage_verified
    assert "coverage unverifiable" in str(pinned.version)


def _midpoint() -> dt.datetime:
    from antlia import history

    symbol, _, _, _ = cached_window()
    stamps = sorted(history.ingests("option_eod", symbol))
    if len(stamps) < 4:
        pytest.skip("store has too few appends to pin between")
    return stamps[len(stamps) // 2]
