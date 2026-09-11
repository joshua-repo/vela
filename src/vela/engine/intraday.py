"""Re-pricing an open book at the session's worst point.

## The problem this exists for

A stop is **path-dependent** and EOD data has no path. Spot can pierce the
short strike at 11am and recover by the close, and a close-only backtest never
sees it: the stop that would have fired in life does not fire in the test.
That makes close-only marking *optimistic for a stop* -- and, symmetrically,
pessimistic for a take-profit, which a resting limit order would have filled
intraday.

The answer is not to pretend one number is the truth. It is to run the
backtest as an **envelope**: two runs bracketing the uncomputable answer.

    close-only   -- the optimistic bound for a stop
    intraday     -- the pessimistic bound

**The band width is the finding.** Narrow, and EOD granularity is not what
limits the result; the conclusions stand. Wide, and there are exactly two
honest responses: redesign the stop to be close-based by construction
("evaluate at 15:55"), which is both backtestable and executable, or buy
intraday quotes for the disputed windows.

## Why the low suffices, and when it does not

A vertical spread's value is monotone in spot, so the day's worst mark for it
occurs exactly at the low -- one synthetic point, no intraday grid needed.
**That shortcut is position-specific.** A straddle or a butterfly is not
monotone; those need both extremes, or the whole range. Nothing here enforces
that, so a non-monotone structure will get a bound that is not a bound.

## Why using our own solved IV is not circular

An implied volatility is a unit conversion of an *observed* market price:
`iv = BS^-1(quote)`. The model is then used only to slide that observed price
to a different spot level. The anchor stays exogenous.

The real subtlety is **causality, not circularity**: today's close IV is
observed *after* today's low. So the reprice uses the **previous** session's
close IV per leg. The bias that leaves is known and one-directional -- vol at
the low was probably higher than the day before, so the stressed loss is
understated, and the pessimistic bound is not pessimistic enough.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence

import numpy as np

from vela.book.position import Position
from vela.data.snapshot import ChainSnapshot
from vela.instrument import Contract
from vela.pricing.base import Greeks, VolModel
from vela.pricing.enrich import YEAR_DAYS


def eligible(positions: Sequence[Position], day: dt.date) -> list[Position]:
    """Positions that existed before `day`'s low happened.

    One opened at today's close could not have stopped out at a price that
    printed hours earlier. Defence in depth rather than the only guard -- a
    position opened today usually has no prior volatility either -- but the
    two protect against different things, and only this one states the rule.
    """
    return [p for p in positions if p.opened_on < day]


def leg_ivs(
    chain: ChainSnapshot, greeks: Greeks, positions: Sequence[Position]
) -> dict[Contract, float]:
    """Today's solved IV for every open leg, for use as tomorrow's prior."""
    out: dict[Contract, float] = {}
    if len(greeks) != len(chain):
        return out
    for position in positions:
        for leg in position.legs:
            row = chain.locate(leg.contract)
            if row is None:
                continue
            value = float(greeks.iv[row])
            if math.isfinite(value) and value > 0:
                out[leg.contract] = value
    return out


def stress_chain(
    chain: ChainSnapshot,
    positions: Sequence[Position],
    *,
    low: float,
    day: dt.date,
    prior_iv: Mapping[Contract, float],
    model: VolModel,
    rate: float,
    dividend_yield: float,
) -> ChainSnapshot:
    """Today's chain with the open legs repriced at `low`.

    The **observed spread is preserved** around the synthetic mid, so the fill
    model keeps charging what the market was actually quoting. Only the level
    moves.

    A leg with no causally usable prior IV is left at its observed quote --
    conservatively unstressed. Silently substituting today's IV would import
    the lookahead this whole module exists to avoid.
    """
    bid = chain.bid.copy()
    ask = chain.ask.copy()

    for position in positions:
        for leg in position.legs:
            row = chain.locate(leg.contract)
            vol = prior_iv.get(leg.contract)
            if row is None or vol is None:
                continue
            years = (leg.contract.expiration - day).days / YEAR_DAYS
            if years <= 0:
                continue
            synthetic = model.price(
                low, leg.contract.strike, years, rate, vol, leg.contract.right, dividend_yield
            )
            half = _observed_half_spread(chain, row)
            bid[row] = max(synthetic - half, 0.0)
            ask[row] = synthetic + half

    return ChainSnapshot(
        date=chain.date,
        symbol=chain.symbol,
        expiration=chain.expiration,
        strike=chain.strike,
        right=chain.right,
        bid=bid,
        ask=ask,
        close=chain.close,
        volume=chain.volume,
    )


def _observed_half_spread(chain: ChainSnapshot, row: int) -> float:
    quoted = float(chain.ask[row]) - float(chain.bid[row])
    return quoted / 2 if np.isfinite(quoted) and quoted > 0 else 0.0
