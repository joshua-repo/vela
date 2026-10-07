"""Equity-curve statistics.

Computed on the **marked** curve, which is realisable value rather than the
mid -- so drawdown here is a drawdown you could actually have suffered, not
one softened by valuing every open position at the midpoint of a spread it
would have had to cross.

Annualisation uses trading sessions rather than calendar days, and the count
comes from the data rather than a constant: a run over a partially covered
window has fewer sessions per year than 252, and pretending otherwise inflates
every ratio.

**Ratios are measured against cash, not against zero.** `risk_free` is the
rate the engine credited to cash, and each session's excess return is the
account's return less what that rate paid over the same calendar gap. Once the
book earns interest this is not optional: an account that is mostly idle cash
would otherwise post a Sharpe made of nothing but the coupon.
`risk_free_return` is the same account left entirely in cash, and
`excess_return` is the difference -- what the strategy added over doing
nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vela.engine.results import EquityPoint

SESSIONS_PER_YEAR = 252

#: Below this, a volatility is floating-point residue, not risk. An account
#: that never trades and only earns its coupon has an excess return of zero to
#: within rounding, and dividing that residue by itself is not a Sharpe ratio.
_NO_RISK = 1e-12


@dataclass(frozen=True, slots=True)
class CurveStats:
    sessions: int
    initial: float
    final: float
    total_return: float
    annualised_return: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_sessions: int
    #: The annual rate cash earned, and the benchmark the ratios use.
    risk_free: float = 0.0
    #: The opening account left in cash over the window, compounded per gap.
    risk_free_return: float = 0.0
    #: `total_return - risk_free_return`.
    excess_return: float = 0.0

    @property
    def max_drawdown_pct(self) -> float:
        return self.max_drawdown


def curve_stats(
    equity: Sequence[EquityPoint],
    *,
    initial: float | None = None,
    risk_free: float = 0.0,
    sessions_per_year: int = SESSIONS_PER_YEAR,
) -> CurveStats:
    """Statistics for a marked equity curve.

    `initial` is the account before the first session traded. It matters:
    the first equity point is recorded *after* that session's fills, so
    measuring from it silently drops day one's P&L from every return figure.
    Defaults to the first point, which is right only for a curve that opens
    flat.

    `risk_free` must be the rate the run actually credited, so that the
    benchmark accrues exactly as the account did: nothing over the opening
    step, then ACT/365 over each gap between sessions.
    """
    if len(equity) < 2:
        raise ValueError("an equity curve needs at least two points")

    marked = [point.equity for point in equity]
    opening = float(marked[0]) if initial is None else float(initial)
    values = np.asarray([opening, *marked], dtype="float64")
    final = float(values[-1])
    years = len(marked) / sessions_per_year

    returns = np.diff(values) / values[:-1]
    gaps = np.asarray(
        [0, *((b.date - a.date).days for a, b in zip(equity, equity[1:], strict=False))],
        dtype="float64",
    )
    carry = risk_free * gaps / 365.0
    excess = returns - carry
    risk_free_return = float(np.prod(1.0 + carry)) - 1.0

    root = math.sqrt(sessions_per_year)
    volatility = float(returns.std(ddof=1)) * root
    excess_volatility = float(excess.std(ddof=1)) * root
    mean = float(excess.mean()) * sessions_per_year

    downside = excess[excess < 0]
    downside_dev = float(downside.std(ddof=1)) * root if len(downside) > 1 else 0.0

    peaks = np.maximum.accumulate(values)
    drawdowns = values / peaks - 1
    trough = int(np.argmin(drawdowns))
    peak = int(np.argmax(values[: trough + 1])) if trough else 0

    total_return = final / opening - 1 if opening else math.nan
    return CurveStats(
        sessions=len(marked),
        initial=opening,
        final=final,
        total_return=total_return,
        annualised_return=(final / opening) ** (1 / years) - 1
        if opening > 0 and years
        else math.nan,
        volatility=volatility,
        sharpe=mean / excess_volatility if excess_volatility > _NO_RISK else math.nan,
        sortino=mean / downside_dev if downside_dev > _NO_RISK else math.nan,
        max_drawdown=float(drawdowns.min()),
        max_drawdown_sessions=trough - peak,
        risk_free=risk_free,
        risk_free_return=risk_free_return,
        excess_return=total_return - risk_free_return,
    )
