"""Per-trade statistics, with their uncertainty attached.

A backtest over one symbol and one year produces tens of trades, not
thousands. At that size a mean is a weak claim, and reporting it alone invites
a decision it cannot support -- so every headline number here carries the
thing that says how much to trust it: a standard error on expectancy, a t
statistic, and a Wilson interval on the win rate.

The Wilson interval rather than the textbook normal one because the normal
interval is wrong exactly where options results live: small n, and a win rate
near 1. A 76% win rate over 42 trades gets `0.61 .. 0.87` from Wilson and an
interval running past 1.0 from the normal approximation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from vela.engine.results import Trade
from vela.execution.fill import Costs

#: 1.96 -- the two-sided 95% normal quantile.
Z95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class TradeStats:
    n: int
    wins: int
    win_rate: float
    win_rate_ci: tuple[float, float]
    expectancy: float
    expectancy_se: float
    t_stat: float
    profit_factor: float
    total_pnl: float
    #: What execution took, split by where it went. Reported next to
    #: `total_pnl` because on a defined-risk options strategy the two are
    #: routinely the same order of magnitude, and a net figure alone cannot
    #: tell a signal problem from an execution assumption.
    costs: Costs
    gross_pnl: float
    mean_return_on_risk: float
    median_return_on_risk: float
    best: float
    worst: float
    mean_held_days: float

    @property
    def significant(self) -> bool:
        """Whether expectancy is two standard errors from zero.

        Not a licence to trade -- an in-sample t statistic over a single
        symbol is a weak claim however large it is. It is a floor: below it,
        the result is indistinguishable from noise and nothing else in this
        object is worth arguing about.
        """
        return abs(self.t_stat) >= 2.0


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion. `(0, 1)` when there is no data."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z / denominator * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def trade_stats(trades: Sequence[Trade]) -> TradeStats:
    if not trades:
        return TradeStats(
            n=0,
            wins=0,
            win_rate=math.nan,
            win_rate_ci=(0.0, 1.0),
            expectancy=math.nan,
            expectancy_se=math.nan,
            t_stat=math.nan,
            profit_factor=math.nan,
            total_pnl=0.0,
            costs=Costs(),
            gross_pnl=0.0,
            mean_return_on_risk=math.nan,
            median_return_on_risk=math.nan,
            best=math.nan,
            worst=math.nan,
            mean_held_days=math.nan,
        )

    pnls = [t.pnl for t in trades]
    n = len(pnls)
    wins = sum(p > 0 for p in pnls)
    mean = sum(pnls) / n
    # Sample standard deviation; a single trade has no dispersion to report.
    variance = sum((p - mean) ** 2 for p in pnls) / (n - 1) if n > 1 else math.nan
    se = math.sqrt(variance / n) if n > 1 else math.nan

    gains = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)

    rors = sorted(t.return_on_margin for t in trades if not math.isnan(t.return_on_margin))

    costs = Costs()
    for t in trades:
        costs = costs + t.costs

    return TradeStats(
        n=n,
        wins=wins,
        win_rate=wins / n,
        win_rate_ci=wilson(wins, n),
        expectancy=mean,
        expectancy_se=se,
        t_stat=mean / se if se and se > 0 else math.nan,
        profit_factor=gains / losses if losses > 0 else math.inf,
        total_pnl=sum(pnls),
        costs=costs,
        gross_pnl=sum(pnls) + costs.total,
        mean_return_on_risk=sum(rors) / len(rors) if rors else math.nan,
        median_return_on_risk=_median(rors),
        best=max(pnls),
        worst=min(pnls),
        mean_held_days=sum(t.held_days for t in trades) / n,
    )


def _median(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2
