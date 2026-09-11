"""The judgements antlia deliberately leaves to its consumers.

antlia's rule is that `raw/` holds what the vendor said, unrepaired, and that
"a crossed quote or a mid below intrinsic is a judgement call that belongs to
the consumer". This module is where Vela makes those calls.

The corporate-action guard exists because of a specific, measured hazard.
Prices are **unadjusted**, antlia ships no corporate-actions table, and
ThetaData 1.0.x exposes no endpoint for one. Across NVDA's 10:1 split on
2024-06-10 -- inside the free plan's own window -- the series does this:

    NVDA  2024-06-07  close 1208.88
    NVDA  2024-06-10  close  121.79

and *nothing warns you*. A backtest that runs over it produces a return series
that is wrong and completely plausible. So the guard's job is not to classify
perfectly; it is to **stop and make a human look**, because a loud failure is
strictly better than a quiet wrong number.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from vela.data.snapshot import ChainSnapshot, SpotBar

#: Price ratios (new/old) produced by splits that actually happen. Forward
#: splits divide the price; reverse splits multiply it.
_SPLIT_FACTORS = (1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30)
_SPLIT_RATIOS = tuple(sorted({1 / f for f in _SPLIT_FACTORS} | {float(f) for f in _SPLIT_FACTORS}))


#: How far the 17:15 NBBO mid may sit from the 16:00 close before a session is
#: worth flagging. Measured on a year of a large-cap underlying, 248 of 251
#: sessions came in under 1%; the exceptions were earnings dates, several
#: percent out.
DIVERGENCE = 0.01


class GuardError(Exception):
    """Base for a refusal to run on data that cannot be trusted."""


class CoverageIncomplete(GuardError):
    """The store cannot vouch for the requested window.

    Distinct from antlia's `NotCovered`, which fires when rows are absent. This
    one also catches the case a read cannot: **a stale expiration listing.**
    antlia plans option coverage from the listing, so a listing fetched before
    the window's end does not know about expirations listed since -- and an
    expiration it never heard of cannot appear as a gap. The read succeeds,
    returns rows, and is quietly missing contracts.

    antlia states the rule exactly: a store warmed in March vouches for a
    January backtest and does not vouch for a September one. Nothing in the
    read path enforces it, so this does.
    """


class CorporateActionSuspected(GuardError):
    """An unadjusted price series appears to cross a split or similar event."""


@dataclass(frozen=True, slots=True)
class GuardFinding:
    kind: str
    date: dt.date
    detail: str

    def __str__(self) -> str:
        return f"{self.date} {self.kind}: {self.detail}"


def _nearest_split_ratio(ratio: float) -> float | None:
    """The split ratio `ratio` looks like, within 3%, or None."""
    for candidate in _SPLIT_RATIOS:
        if abs(ratio / candidate - 1) < 0.03:
            return candidate
    return None


def check_corporate_actions(
    bars: Sequence[SpotBar],
    *,
    chains: Mapping[dt.date, ChainSnapshot] | None = None,
    threshold: float = 0.30,
    raises: bool = True,
) -> list[GuardFinding]:
    """Findings for every session-over-session move large enough to be a split.

    `threshold` is a *fraction*, applied to `close_t / close_{t-1} - 1`. The
    default of 0.30 is chosen so the guard fires on every real split (the
    smallest common one, 3:2, moves price by 33%) and stays silent on ordinary
    volatility. It will also fire on a genuine 30%+ single-day move -- which
    is the correct outcome: that too is something a human should look at
    before a year of backtest results rests on it.

    When `chains` is given the answer stops being a heuristic. A split adjusts
    the *contract terms* as well as the price, so the whole listed strike grid
    rescales by the same factor; an ordinary crash moves the price and leaves
    the grid where it was. That cross-check is what separates the two.

    Raises `CorporateActionSuspected` on the first finding unless
    `raises=False`, which is how a caller asks to inspect rather than run.
    """
    findings: list[GuardFinding] = []
    for prev, cur in zip(bars, bars[1:], strict=False):
        if prev.close <= 0 or cur.close <= 0:
            continue
        ratio = cur.close / prev.close
        if abs(ratio - 1) <= threshold:
            continue

        split = _nearest_split_ratio(ratio)
        detail = (
            f"close {prev.close:g} ({prev.date}) -> {cur.close:g}, "
            f"a {(ratio - 1) * 100:+.1f}% session-over-session move"
        )
        if split is not None:
            detail += f"; ratio {ratio:.4f} matches a {_describe(split)} split"
        else:
            detail += f"; ratio {ratio:.4f} matches no common split ratio"

        confirmation = _confirm_with_strikes(chains, prev.date, cur.date, ratio)
        if confirmation:
            detail += f"; {confirmation}"

        findings.append(GuardFinding("corporate_action", cur.date, detail))

    if findings and raises:
        raise CorporateActionSuspected(
            "prices are unadjusted and antlia ships no corporate-actions table, so "
            "this window cannot be backtested as-is:\n  "
            + "\n  ".join(str(f) for f in findings)
            + "\nNarrow the window past the event, or pass raises=False to inspect."
        )
    return findings


def _describe(ratio: float) -> str:
    return f"{1 / ratio:g}:1" if ratio < 1 else f"1:{ratio:g}"


def _confirm_with_strikes(
    chains: Mapping[dt.date, ChainSnapshot] | None,
    before: dt.date,
    after: dt.date,
    price_ratio: float,
) -> str | None:
    """Whether the listed strike grid rescaled along with the price.

    A split adjusts the contracts too, so the grid moves by the same factor.
    A crash does not touch it.
    """
    if chains is None:
        return None
    a, b = chains.get(before), chains.get(after)
    if a is None or b is None or not len(a) or not len(b):
        return None
    strike_ratio = float(np.median(b.strike)) / float(np.median(a.strike))
    if not math.isfinite(strike_ratio) or strike_ratio <= 0:
        return None
    if abs(strike_ratio / price_ratio - 1) < 0.15:
        return (
            f"the strike grid rescaled with it ({strike_ratio:.4f} vs price {price_ratio:.4f}) "
            "-- this is a corporate action, not a market move"
        )
    return (
        f"the strike grid did NOT rescale ({strike_ratio:.4f} vs price {price_ratio:.4f}) "
        "-- this looks like a real market move"
    )


def quote_divergence(
    bars: Sequence[SpotBar], *, threshold: float = DIVERGENCE
) -> list[GuardFinding]:
    """Sessions where the 17:15 NBBO mid disagrees with the 16:00 close.

    Not an error -- an annotation. Measured on a year of a large-cap
    underlying the two agree to within 1% on 248 of 251 sessions; the
    exceptions were earnings dates, where the post-close quote already carries
    the reaction. Flagging them is what lets a result say "this trade was
    entered on an earnings day" instead of burying it.
    """
    findings = []
    for bar in bars:
        d = bar.divergence
        if math.isfinite(d) and abs(d) > threshold:
            findings.append(
                GuardFinding(
                    "quote_divergence",
                    bar.date,
                    f"close {bar.close:g} vs 17:15 mid {bar.mid:g} ({d * 100:+.2f}%)",
                )
            )
    return findings
