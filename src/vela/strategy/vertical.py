"""Bullish vertical spreads, credit and debit, as one structure.

A bull put spread and a bull call spread are the **same shape**: sell the
higher strike, buy the one `width` below it. Only the right differs -- and
with it whether the structure is a credit (puts, struck below spot) or a debit
(calls, struck around or above it). Writing them as two classes would
duplicate every selection and exit rule to express that one difference.

    puts,  moneyness 0.97  ->  short 97% put,  long 5 lower   -- credit
    calls, moneyness 1.03  ->  short 103% call, long 5 lower  -- debit

## Each exit is a fraction of its own extreme

`take_profit` is a fraction of the structure's **max profit**; `stop_loss` a
fraction of its **max loss**. Both come from the terminal payoff, so both mean
the intuitive thing for a credit and a debit alike: "close once I have taken
half of the best case", "close once I have given up half of the worst".

Measuring both against capital at risk -- the obvious first choice -- is
subtly broken, and the first real-data run proved it. A $5-wide put spread
sold for $1.90 ties up $810 and can never make more than $190, so a
take-profit of "0.5 x risk" asks for $405 and never fires. The backtest ran
clean, closed all 31 trades on the time exit, and reported a profit. Nothing
looked wrong.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np

from vela.book.position import Position
from vela.data.snapshot import ChainSnapshot
from vela.engine.context import Context
from vela.instrument import Contract, Right
from vela.orders import Close, LegOrder, Open, Order
from vela.strategy.base import Strategy
from vela.strategy.gate import AlwaysOpen, Gate


@dataclass(slots=True)
class VerticalSpread(Strategy):
    """Sell a vertical, hold it to a target, a stop, a time exit or expiry.

    Signal parameters carry **no defaults**. A committed default is a live
    parameter value in shared code, and the moment one exists every run that
    omits it silently agrees to somebody's guess. Only structural knobs -- how
    many may be open, how far apart entries must be -- have them.
    """

    right: Right
    moneyness: float
    width: float
    entry_dte: tuple[int, int]
    take_profit: float
    stop_loss: float
    exit_dte: int | None = None
    #: When to allow an entry at all. The default has no signal -- it is the
    #: null baseline, and a gated variant that does not beat it has shown
    #: nothing. See `vela.strategy.gate`.
    gate: Gate = field(default_factory=AlwaysOpen)
    max_concurrent: int = 1
    min_days_between_entries: int = 0
    tag: str = "vertical"
    name: str = field(default="VerticalSpread", init=False)
    _last_entry: dt.date | None = field(default=None, init=False, repr=False)
    _last_day: dt.date | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive -- it is the distance between strikes")
        lo, hi = self.entry_dte
        if lo > hi:
            raise ValueError(f"entry_dte is empty: {self.entry_dte}")
        gate = "" if isinstance(self.gate, AlwaysOpen) else f" {self.gate.name}"
        self.name = f"Vertical({self.right} {self.moneyness:g}x w{self.width:g}{gate})"

    def on_day(self, ctx: Context) -> list[Order]:
        # A day that does not advance means this instance is being reused for
        # a second run; the spacing rule would otherwise carry over from the
        # last one. A fresh instance per run is the rule -- this makes reuse
        # correct rather than quietly wrong. Stress passes share the day and
        # so do not trip it.
        if self._last_day is not None and ctx.day < self._last_day:
            self._last_entry = None
        self._last_day = ctx.day

        # Unconditionally, and before anything else: a gate's history has to
        # advance on days it is shut, or its own range is computed from the
        # days it happened to be open.
        self.gate.observe(ctx)

        orders: list[Order] = [*self._exits(ctx)]
        entry = self._entry(ctx) if self._may_enter(ctx, len(orders)) else None
        if entry is not None:
            orders.append(entry)
            self._last_entry = ctx.day
        return orders

    # -- exits -------------------------------------------------------------

    def _exits(self, ctx: Context) -> list[Order]:
        out: list[Order] = []
        for position in ctx.book.positions:
            if position.tag != self.tag:
                continue
            reason = self._exit_reason(ctx, position)
            if reason is not None:
                out.append(Close(position.id, reason=reason))
        return out

    def _exit_reason(self, ctx: Context, position: Position) -> str | None:
        pnl = ctx.unrealised_at_mid(position)

        target = position.max_profit
        if math.isfinite(target) and target > 0 and pnl >= self.take_profit * target:
            return "take_profit"

        risk = position.max_loss
        if math.isfinite(risk) and risk > 0 and pnl <= -self.stop_loss * risk:
            return "stop_loss"

        if self.exit_dte is not None and (position.expiration - ctx.day).days <= self.exit_dte:
            return "time_exit"
        return None

    # -- entry -------------------------------------------------------------

    def _may_enter(self, ctx: Context, pending_closes: int) -> bool:
        if ctx.stressed:
            # A stressed context is a hypothetical at the session's low. Only a
            # stop is acted on there, so an entry would be silently discarded
            # -- but it would still advance `_last_entry` and skew the spacing
            # rule. Decline explicitly.
            return False
        # Closes issued today free their slot today: the engine executes every
        # close before any open, so counting them here matches what happens.
        open_after = len(ctx.book.by_tag(self.tag)) - pending_closes
        if open_after >= self.max_concurrent:
            return False
        too_soon = (
            self._last_entry is not None
            and (ctx.day - self._last_entry).days < self.min_days_between_entries
        )
        if too_soon:
            return False
        return self.gate.is_open(ctx)

    def _entry(self, ctx: Context) -> Open | None:
        spot = ctx.spot.close
        candidates = ctx.chain.select(right=self.right, dte=self.entry_dte, quoted=True)
        if not len(candidates):
            return None

        expiry = self._pick_expiration(candidates, ctx.day)
        rung = candidates.select(expiration=expiry)

        short = self._nearest_strike(rung, self.moneyness * spot)
        if short is None:
            return None
        long = self._nearest_strike(rung, short - self.width, below=short)
        if long is None:
            return None

        return Open(
            legs=(
                LegOrder(Contract(ctx.symbol, expiry, short, self.right), -1),
                LegOrder(Contract(ctx.symbol, expiry, long, self.right), +1),
            ),
            tag=self.tag,
        )

    def _pick_expiration(self, chain: ChainSnapshot, day: dt.date) -> dt.date:
        """The expiration nearest the middle of the entry window.

        Nearest the middle rather than the nearest available, so a roll does
        not make the sampled maturity sawtooth between the window's edges --
        the artefact that contaminated the prototype's IV series.
        """
        lo, hi = self.entry_dte
        target = (lo + hi) / 2
        return min(chain.expirations(), key=lambda e: abs((e - day).days - target))

    @staticmethod
    def _nearest_strike(
        chain: ChainSnapshot, want: float, below: float | None = None
    ) -> float | None:
        """The listed strike closest to `want`, optionally strictly below `below`.

        Closest-available rather than exact, because a strike grid is not
        uniform: $2.50 near the money, $5 further out, and a width that lands
        between rungs would otherwise silently decline every entry.
        """
        strikes = chain.strike if below is None else chain.strike[chain.strike < below]
        if not len(strikes):
            return None
        return float(strikes[int(np.argmin(np.abs(strikes - want)))])
