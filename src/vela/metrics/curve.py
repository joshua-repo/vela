"""Equity-curve statistics.

Computed on the **marked** curve, which is realisable value rather than the
mid -- so drawdown here is a drawdown you could actually have suffered, not
one softened by valuing every open position at the midpoint of a spread it
would have had to cross.

Annualisation uses trading sessions rather than calendar days, and the count
comes from the data rather than a constant: a run over a partially covered
window has fewer sessions per year than 252, and pretending otherwise inflates
every ratio.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vela.engine.results import EquityPoint

SESSIONS_PER_YEAR = 252


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

    @property
    def max_drawdown_pct(self) -> float:
        return self.max_drawdown


def curve_stats(
    equity: Sequence[EquityPoint],
    *,
    initial: float | None = None,
    sessions_per_year: int = SESSIONS_PER_YEAR,
) -> CurveStats:
    """Statistics for a marked equity curve.

    `initial` is the account before the first session traded. It matters:
    the first equity point is recorded *after* that session's fills, so
    measuring from it silently drops day one's P&L from every return figure.
    Defaults to the first point, which is right only for a curve that opens
    flat.
    """
    if len(equity) < 2:
        raise ValueError("an equity curve needs at least two points")

    marked = [point.equity for point in equity]
    opening = float(marked[0]) if initial is None else float(initial)
    values = np.asarray([opening, *marked], dtype="float64")
    final = float(values[-1])
    years = len(marked) / sessions_per_year

    returns = np.diff(values) / values[:-1]
    volatility = float(returns.std(ddof=1)) * math.sqrt(sessions_per_year)
    mean = float(returns.mean()) * sessions_per_year

    downside = returns[returns < 0]
    downside_dev = (
        float(downside.std(ddof=1)) * math.sqrt(sessions_per_year) if len(downside) > 1 else 0.0
    )

    peaks = np.maximum.accumulate(values)
    drawdowns = values / peaks - 1
    trough = int(np.argmin(drawdowns))
    peak = int(np.argmax(values[: trough + 1])) if trough else 0

    return CurveStats(
        sessions=len(marked),
        initial=opening,
        final=final,
        total_return=final / opening - 1 if opening else math.nan,
        annualised_return=(final / opening) ** (1 / years) - 1
        if opening > 0 and years
        else math.nan,
        volatility=volatility,
        sharpe=mean / volatility if volatility > 0 else math.nan,
        sortino=mean / downside_dev if downside_dev > 0 else math.nan,
        max_drawdown=float(drawdowns.min()),
        max_drawdown_sessions=trough - peak,
    )
