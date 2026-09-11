"""Entry gates: whether today is a day to open at all.

A seam, and the reason pricing exists. `VerticalSpread` on its own has **no
signal** -- it enters on a schedule, every N days, whatever the market is
doing. Sweeping the parameters of a scheduled strategy optimises noise, which
is what the first grid demonstrated: eighteen combinations, best t-statistic
+0.07.

A gate is what makes an entry conditional on the world. Keeping it separate
from the structure means the same gate drives a vertical, a strangle or a
condor, and -- more useful -- that `AlwaysOpen` is the **null baseline** every
gated result must beat. A gated strategy that does not beat its own
always-enter null has not demonstrated a signal, however good its P&L looks.

## Why the gate reads yesterday

Implied volatility here is solved from the 17:15 post-close quotes (see
`vela.pricing.enrich`). Reading today's IV to decide today's entry is
intraday lookahead, and on an earnings date it is enormous -- the quote
already contains the reaction. `IVRank` therefore observes the **previous**
session and never touches the current one.
"""

from __future__ import annotations

import datetime as dt
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from vela.engine.context import Context
from vela.pricing.series import atm_iv

#: A window narrower than this fraction of its own level is not a range.
#: One volatility point on a 25% level is 4%, so this is far below anything
#: a market produces and well above float noise.
_FLAT = 1e-6


class Gate(ABC):
    """May a strategy open a position today?"""

    name: str = "Gate"

    def observe(self, ctx: Context) -> None:  # noqa: B027 -- an opt-in hook
        """Called every session, before `is_open`, so state stays current.

        A no-op by default rather than abstract: a stateless gate has nothing
        to remember, and making every one of them write an empty method would
        be ceremony.

        Separate from `is_open` because a gate's history must advance on days
        the strategy cannot trade -- when the book is full, or when the gate
        itself is shut. The prototype had no such hook and worked around it by
        calling its entry check every day and discarding the answer.
        """

    @abstractmethod
    def is_open(self, ctx: Context) -> bool:
        """Whether an entry is permitted today."""


class AlwaysOpen(Gate):
    """No signal at all. The null every gated result is measured against."""

    name = "AlwaysOpen"

    def is_open(self, ctx: Context) -> bool:
        return True


@dataclass(slots=True)
class IVRank(Gate):
    """Open only when at-the-money IV is high within its own recent range.

    Rank rather than level, because an absolute threshold does not transfer:
    on a measured large-cap year, at-the-money IV never approached the
    "IV > 50%" rule of thumb, and only junk wing strikes ever did. A rank is
    scale-free and says what was actually meant -- *expensive for this
    underlying, lately*.

    `minimum` is a fraction of the observed range: 0.5 means "in the upper
    half of the last `lookback` sessions". `warmup` sessions must be observed
    before the gate opens at all, because a range computed from four
    observations is not a range.
    """

    minimum: float
    lookback: int = 120
    warmup: int = 20
    target_dte: int = 37
    band: float = 0.05
    name: str = field(default="IVRank", init=False)
    _history: list[float] = field(default_factory=list, init=False, repr=False)
    _seen: set[dt.date] = field(default_factory=set, init=False, repr=False)
    _last_day: dt.date | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum <= 1.0:
            raise ValueError("minimum is a rank in 0..1, not a volatility")
        if self.warmup < 2:
            raise ValueError("a range needs at least two observations")
        self.name = f"IVRank(>={self.minimum:g})"

    def observe(self, ctx: Context) -> None:
        # Sessions advance monotonically inside a run, so a day that is not
        # after the last one means this instance is being reused for a second
        # run. Reuse used to be silently wrong rather than loudly wrong: the
        # deduplication below would skip every already-seen session, the
        # history would never advance, and the rank would freeze at the *last*
        # value of the previous run -- pinning the gate open or shut for the
        # whole of the new one. Found by reusing a gate across an envelope
        # pair and getting a "gated" run with every trade in it.
        # Strictly earlier, not "not later": the intraday stress pass re-asks
        # the strategy on the *same* day, and treating that as a restart would
        # clear the history on every stressed session.
        if self._last_day is not None and ctx.day < self._last_day:
            self._history.clear()
            self._seen.clear()
        self._last_day = ctx.day

        previous = self._previous_session(ctx)
        if previous is None or previous in self._seen:
            return
        self._seen.add(previous)

        bar = ctx.market.bar(previous)
        spot = bar.mid if math.isfinite(bar.mid) else bar.close
        value = atm_iv(
            ctx.market.chain_on(previous),
            ctx.market.greeks_on(previous),
            spot,
            target_dte=self.target_dte,
            band=self.band,
        )
        if math.isfinite(value):
            self._history.append(value)

    def is_open(self, ctx: Context) -> bool:
        return self.rank() >= self.minimum

    def rank(self) -> float:
        """Today's rank in 0..1, or NaN before warmup or on a degenerate range.

        "Degenerate" is a *relative* test, not `high == low`. A window whose
        range is a rounding error still divides, and turns solver noise into a
        confident 0.0 or 1.0 -- the gate would then be driven by the last bit
        of a float. Below `_FLAT` of the level there is no range to rank in.
        """
        if len(self._history) < self.warmup:
            return math.nan
        window = self._history[-self.lookback :]
        low, high = min(window), max(window)
        if high <= low or (high - low) <= _FLAT * abs(high):
            return math.nan
        return (window[-1] - low) / (high - low)

    @property
    def observations(self) -> int:
        return len(self._history)

    @staticmethod
    def _previous_session(ctx: Context) -> dt.date | None:
        recent = ctx.market.history(2)
        return recent[0].date if len(recent) == 2 else None
