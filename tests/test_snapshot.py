"""ChainSnapshot: lookup, screening, and the two-price distinction."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from tests.conftest import bar, chain
from vela.data.snapshot import ChainSnapshot, Quote
from vela.instrument import Contract

EXPIRY = dt.date(2026, 2, 20)
DAY = dt.date(2026, 1, 2)


def test_length_and_derived_columns(simple_chain: ChainSnapshot) -> None:
    assert len(simple_chain) == 10
    assert simple_chain.dte.tolist() == [49] * 10
    # spot 100: the 90 put is 10 OTM, so mid is pure extrinsic
    otm = simple_chain.locate(Contract("TEST", EXPIRY, 90.0, "P"))
    assert otm is not None
    assert simple_chain.mid[otm] == pytest.approx(1.20)

    # ...and the 110 put is 10 ITM: intrinsic 10 plus the same extrinsic
    itm = simple_chain.locate(Contract("TEST", EXPIRY, 110.0, "P"))
    assert itm is not None
    assert simple_chain.mid[itm] == pytest.approx(11.20)


def test_locate_is_exact_and_returns_none_for_absent() -> None:
    snap = chain(DAY, {EXPIRY: [(100.0, "P", 1.0, 1.2)]})
    assert snap.locate(Contract("TEST", EXPIRY, 100.0, "P")) == 0
    assert snap.locate(Contract("TEST", EXPIRY, 100.0, "C")) is None
    assert snap.locate(Contract("TEST", EXPIRY, 101.0, "P")) is None
    assert snap.locate(Contract("TEST", dt.date(2026, 3, 20), 100.0, "P")) is None


def test_locate_tolerates_vendor_float_noise() -> None:
    """antlia stores strikes unrounded; a lookup must still land."""
    snap = chain(DAY, {EXPIRY: [(100.00000000000001, "P", 1.0, 1.2)]})
    assert snap.locate(Contract("TEST", EXPIRY, 100.0, "P")) == 0


def test_select_filters_compose(simple_chain: ChainSnapshot) -> None:
    puts = simple_chain.select(right="P")
    assert len(puts) == 5
    assert set(puts.right.tolist()) == {"P"}

    band = simple_chain.select(right="P", strikes=(95.0, 105.0))
    assert sorted(band.strike.tolist()) == [95.0, 100.0, 105.0]

    assert len(simple_chain.select(dte=(0, 10))) == 0
    assert len(simple_chain.select(dte=(40, 60))) == 10


def test_select_quoted_drops_unquotable_rows() -> None:
    snap = chain(
        DAY,
        {EXPIRY: [(100.0, "P", 1.0, 1.2), (80.0, "P", 0.0, 0.05), (70.0, "P", math.nan, math.nan)]},
    )
    assert len(snap) == 3
    assert len(snap.select(quoted=True)) == 1


def test_marking_must_not_use_the_entry_filter() -> None:
    """A leg going unquoted must stay visible, or a position vanishes silently."""
    snap = chain(DAY, {EXPIRY: [(80.0, "P", 0.0, 0.05)]})
    leg = Contract("TEST", EXPIRY, 80.0, "P")
    assert snap.locate(leg) is not None  # marking still finds it
    assert snap.select(quoted=True).locate(leg) is None  # screening does not


def test_quote_semantics() -> None:
    assert Quote(1.0, 1.2, 1.1, 5).mid == pytest.approx(1.1)
    assert Quote(1.0, 1.2, 1.1, 5).spread == pytest.approx(0.2)

    no_bid = Quote(0.0, 0.05, 0.0, 0)
    assert not no_bid.has_quote
    assert math.isnan(no_bid.mid)
    assert not no_bid.has_trade

    crossed = Quote(1.3, 1.2, 1.25, 3)
    assert not crossed.has_quote


def test_spot_bar_divergence_is_the_earnings_flag() -> None:
    quiet = bar(dt.date(2026, 1, 2), 100.0, bid=99.99, ask=100.01)
    assert abs(quiet.divergence) < 0.001

    earnings = bar(dt.date(2026, 1, 2), 100.0, bid=104.0, ask=104.1)
    assert earnings.divergence == pytest.approx(0.0405, abs=1e-3)


def test_mismatched_column_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="expected"):
        ChainSnapshot(
            date=DAY,
            symbol="TEST",
            expiration=np.asarray([EXPIRY], dtype="datetime64[D]"),
            strike=np.asarray([100.0, 105.0]),
            right=np.asarray(["P", "P"], dtype="<U1"),
            bid=np.asarray([1.0, 1.0]),
            ask=np.asarray([1.2, 1.2]),
            close=np.asarray([1.1, 1.1]),
            volume=np.asarray([1, 1]),
        )
