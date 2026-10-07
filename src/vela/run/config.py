"""One run's configuration, and the only place its defaults live.

The library deliberately has few defaults -- `VerticalSpread` carries none for
its signal, and minimal ones for its structure -- so assembling a run means
choosing values for every knob. Before this module the CLI chose them inline,
and anything else that assembled a run (a test, a notebook, a combo cell)
chose its own: building the "same" spread straight from the library ran with
one position at a time instead of three, no spacing and no time exit, and
produced 19 trades where `vela` produced 42. Nothing said why.

So the run layer has exactly one source: `RunConfig`. The CLI takes its
argparse defaults from these fields, and a combo builds each cell as a
`RunConfig`, so a single run and a grid cell are the same object built two
ways.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, fields
from typing import Any

from vela.data.market import MarketView
from vela.engine.loop import BacktestEngine
from vela.execution.fill import FillModel
from vela.instrument import Right
from vela.pricing.enrich import Enricher
from vela.risk.margin import MaxLossMargin
from vela.risk.sizing import FixedRisk
from vela.strategy.gate import AlwaysOpen, Gate, IVRank
from vela.strategy.vertical import VerticalSpread


@dataclass(frozen=True, slots=True)
class RunConfig:
    # -- data
    symbol: str
    start: str
    end: str
    #: Pin the read to the store as it stood then. None reads the newest.
    as_of: dt.datetime | None = None
    #: Must match the --max-dte the cache was filled with.
    max_dte: int = 45

    # -- structure
    right: Right = "P"
    moneyness: float = 0.95
    width: float = 5.0
    entry_dte: tuple[int, int] = (30, 45)

    # -- exits
    take_profit: float = 0.5
    stop_loss: float = 0.5
    #: Close at this DTE. None holds to the other exits or expiry.
    exit_dte: int | None = 7

    # -- book
    max_concurrent: int = 3
    spacing: int = 7
    cash: float = 100_000.0
    risk: float = 1_000.0

    # -- signal
    #: None is the no-signal null.
    iv_rank: float | None = None
    iv_lookback: int = 120
    iv_warmup: int = 20

    # -- execution and pricing
    haircut: float = 0.5
    commission: float = 0.65
    blackout: float | None = None
    rate: float = 0.04
    dividend_yield: float = 0.0
    #: What cash earns, and the benchmark. None resolves to `rate`: one
    #: risk-free number unless told otherwise.
    cash_rate: float | None = None

    def __post_init__(self) -> None:
        if self.cash_rate is None:
            object.__setattr__(self, "cash_rate", self.rate)

    @property
    def carry(self) -> float:
        """`cash_rate`, resolved. Never None after construction."""
        assert self.cash_rate is not None
        return self.cash_rate

    # -- assembly -------------------------------------------------------------

    def enricher(self) -> Enricher:
        return Enricher(rate=self.rate, dividend_yield=self.dividend_yield)

    def load(self) -> MarketView:
        """Read this configuration's window. Raises what `MarketView.load` does."""
        return MarketView.load(
            self.symbol,
            self.start,
            self.end,
            max_dte=self.max_dte,
            enrich=self.enricher(),
            as_of=self.as_of,
        )

    def engine(self, **overrides: Any) -> BacktestEngine:
        """The engine for this configuration; `overrides` replace fill knobs
        (`haircut`, `commission`) for a probe such as the break-even scan."""
        fill = {"haircut": self.haircut, "commission": self.commission, **overrides}
        return BacktestEngine(
            fill=FillModel(**fill),
            margin=MaxLossMargin(),
            sizing=FixedRisk(self.risk),
            initial_cash=self.cash,
            blackout=self.blackout,
            cash_rate=self.carry,
        )

    def gate(self) -> Gate:
        if self.iv_rank is None:
            return AlwaysOpen()
        return IVRank(minimum=self.iv_rank, lookback=self.iv_lookback, warmup=self.iv_warmup)

    def strategy(self, gate: Gate | None = None) -> VerticalSpread:
        """A **fresh** strategy every call: a gate carries observation history
        that must not leak between runs (Decision 24). `gate` overrides this
        configuration's own -- how a gated run builds its null."""
        return VerticalSpread(
            right=self.right,
            moneyness=self.moneyness,
            width=self.width,
            entry_dte=self.entry_dte,
            take_profit=self.take_profit,
            stop_loss=self.stop_loss,
            exit_dte=self.exit_dte,
            gate=self.gate() if gate is None else gate,
            max_concurrent=self.max_concurrent,
            min_days_between_entries=self.spacing,
            tag="vertical",
        )


#: Every field's default, for a front end that has to state them (argparse).
DEFAULTS: dict[str, Any] = {
    f.name: f.default for f in fields(RunConfig) if f.name not in ("symbol", "start", "end")
}
