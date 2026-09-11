"""The judgements antlia leaves to us: splits, and the post-close quote."""

from __future__ import annotations

import datetime as dt

import pytest

from tests.conftest import bar, chain
from vela.data.guards import (
    CorporateActionSuspected,
    check_corporate_actions,
    quote_divergence,
)
from vela.data.snapshot import SpotBar

DAY0, DAY1 = dt.date(2026, 1, 2), dt.date(2026, 1, 5)
EXPIRY = dt.date(2026, 2, 20)


def series(closes: list[float]) -> list[SpotBar]:
    start = dt.date(2026, 1, 2)
    return [bar(start + dt.timedelta(days=i), c) for i, c in enumerate(closes)]


def test_quiet_series_passes() -> None:
    assert check_corporate_actions(series([100, 101, 99, 103, 98])) == []


def test_ten_to_one_split_is_refused_and_named() -> None:
    with pytest.raises(CorporateActionSuspected) as excinfo:
        check_corporate_actions(series([1208.88, 121.79, 120.91]))
    assert "10:1 split" in str(excinfo.value)


def test_reverse_split_is_caught_too() -> None:
    with pytest.raises(CorporateActionSuspected) as excinfo:
        check_corporate_actions(series([2.0, 20.0, 19.5]))
    assert "1:10 split" in str(excinfo.value)


def test_a_real_crash_still_halts_but_is_described_differently() -> None:
    """A 40% drop is not a split; the guard still stops, and says so."""
    findings = check_corporate_actions(series([100.0, 60.0]), raises=False)
    assert len(findings) == 1
    assert "no common split ratio" in findings[0].detail


def test_raises_false_reports_instead_of_stopping() -> None:
    findings = check_corporate_actions(series([1208.88, 121.79]), raises=False)
    assert [f.kind for f in findings] == ["corporate_action"]


def test_strike_grid_confirms_a_split() -> None:
    """A split adjusts the contracts; the whole grid rescales with the price."""
    before = chain(DAY0, {EXPIRY: [(1000.0, "P", 1.0, 1.2), (1200.0, "P", 1.0, 1.2)]})
    after = chain(DAY1, {EXPIRY: [(100.0, "P", 1.0, 1.2), (120.0, "P", 1.0, 1.2)]})
    findings = check_corporate_actions(
        [bar(DAY0, 1208.88), bar(DAY1, 121.79)],
        chains={DAY0: before, DAY1: after},
        raises=False,
    )
    assert "strike grid rescaled with it" in findings[0].detail


def test_strike_grid_exonerates_a_crash() -> None:
    """Price collapsed, contracts did not -- that is a market move."""
    grid = [(100.0, "P", 1.0, 1.2), (120.0, "P", 1.0, 1.2)]
    findings = check_corporate_actions(
        [bar(DAY0, 100.0), bar(DAY1, 60.0)],
        chains={DAY0: chain(DAY0, {EXPIRY: grid}), DAY1: chain(DAY1, {EXPIRY: grid})},
        raises=False,
    )
    assert "did NOT rescale" in findings[0].detail
    assert "real market move" in findings[0].detail


def test_quote_divergence_flags_only_the_outliers() -> None:
    bars = [
        bar(DAY0, 100.0, bid=99.99, ask=100.01),  # ordinary session
        bar(DAY1, 100.0, bid=104.0, ask=104.1),  # post-close reaction
    ]
    findings = quote_divergence(bars)
    assert [f.date for f in findings] == [DAY1]
    assert "+4.05%" in findings[0].detail
