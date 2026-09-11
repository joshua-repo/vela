"""What a run produces.

`EquityPoint` deliberately carries more than the equity number: cash, market
value, realised and unrealised P&L are all recorded per session, so the cash
invariant

    equity - initial_cash == realised + unrealised

is checkable **from the result alone**, for every day of the run, without
re-running anything. An invariant you can only assert from inside the loop is
an invariant nobody checks.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from vela.book.position import Leg, Position
from vela.execution.fill import Costs


@dataclass(frozen=True, slots=True)
class Trade:
    """One position, from open to close."""

    position_id: int
    tag: str
    legs: tuple[Leg, ...]
    units: int
    opened_on: dt.date
    closed_on: dt.date
    expiration: dt.date
    entry_cash: float
    exit_cash: float
    entry_margin: float
    reason: str
    costs: Costs = Costs()

    @property
    def pnl(self) -> float:
        """Total signed cash this position produced over its life."""
        return self.entry_cash + self.exit_cash

    @property
    def gross_pnl(self) -> float:
        """P&L before execution took its share.

        Reported alongside `pnl` because on a defined-risk options strategy
        the gap between the two is routinely larger than either -- and a
        reader who cannot see the gap cannot tell whether they are looking at
        a signal problem or an execution assumption.
        """
        return self.pnl + self.costs.total

    @property
    def held_days(self) -> int:
        return (self.closed_on - self.opened_on).days

    @property
    def return_on_margin(self) -> float:
        """P&L per dollar of capital tied up -- size-invariant, unlike raw P&L.

        NaN-free by construction: a structure that ties up nothing has no
        return on capital to report, and 0.0 would read as "broke even".
        """
        return self.pnl / self.entry_margin if self.entry_margin > 0 else float("nan")


@dataclass(frozen=True, slots=True)
class EquityPoint:
    date: dt.date
    spot: float
    cash: float
    market_value: float
    realised: float
    unrealised: float
    open_positions: int

    @property
    def equity(self) -> float:
        return self.cash + self.market_value


@dataclass(frozen=True, slots=True)
class DataQuality:
    """What the run noticed about the data it ran on.

    Every field here is something that was previously true-and-invisible: the
    checks existed as functions nobody called, so "checked, and clean" looked
    exactly like "never checked". A reported zero is evidence; an absent
    number is not.
    """

    sessions: int
    #: Sessions with an underlying bar but no option rows at all. Positions
    #: still mark -- at intrinsic -- so this is silent unless counted.
    sessions_without_chain: int
    #: Leg-days the book marked, and how many of those had no two-sided quote
    #: and fell back to intrinsic. A position marked at intrinsic for a stretch
    #: is a data problem wearing a P&L costume.
    marked_leg_days: int
    unquoted_leg_days: int
    #: Sessions where the 17:15 NBBO mid disagrees with the 16:00 close by more
    #: than the guard's threshold -- in practice, earnings dates.
    divergent_sessions: int
    #: Trades opened or closed on one of those. The engine decides on the
    #: 16:00 close, but selects and prices legs off the 17:15 chain, which on
    #: an earnings date already contains the reaction. Not repairable with EOD
    #: data; the honest response is to report it.
    decisions_on_divergent_sessions: int

    @property
    def unquoted_rate(self) -> float:
        return self.unquoted_leg_days / self.marked_leg_days if self.marked_leg_days else 0.0


@dataclass(frozen=True, slots=True)
class Declined:
    """An order the engine would not fill, and why.

    Kept because "the strategy tried to enter 40 times and could afford 12" is
    a fact about the result, not a log line. A run that silently drops entries
    reports a strategy nobody ran.
    """

    date: dt.date
    tag: str
    reason: str


@dataclass(slots=True)
class Result:
    initial_cash: float
    trades: list[Trade] = field(default_factory=list)
    equity: list[EquityPoint] = field(default_factory=list)
    declined: list[Declined] = field(default_factory=list)
    #: Positions still held on the final session. They never became trades, so
    #: `realised` excludes them -- their P&L lives in the last point's
    #: `unrealised`. Reporting only realised P&L on a run that ends with an
    #: open book quietly omits part of the answer.
    open_at_end: tuple[Position, ...] = ()
    quality: DataQuality | None = None

    @property
    def final_equity(self) -> float:
        return self.equity[-1].equity if self.equity else self.initial_cash

    @property
    def total_pnl(self) -> float:
        return self.final_equity - self.initial_cash

    @property
    def realised(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def costs(self) -> Costs:
        """What execution took across every closed trade."""
        total = Costs()
        for trade in self.trades:
            total = total + trade.costs
        return total

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return float("nan")
        return sum(t.pnl > 0 for t in self.trades) / len(self.trades)

    def invariant_breaks(self, tolerance: float = 1e-6) -> list[EquityPoint]:
        """Sessions where `equity - initial != realised + unrealised`.

        Empty is the only acceptable answer. A break means cash moved without
        a position explaining it, which is the failure mode a backtest hides
        best -- the number still looks plausible.
        """
        return [
            point
            for point in self.equity
            if abs((point.equity - self.initial_cash) - (point.realised + point.unrealised))
            > tolerance
        ]

    def __str__(self) -> str:
        return (
            f"<Result trades={len(self.trades)} declined={len(self.declined)} "
            f"pnl={self.total_pnl:+,.2f} final={self.final_equity:,.2f}>"
        )
