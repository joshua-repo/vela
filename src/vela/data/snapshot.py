"""One day of market data, laid out columnar.

Nothing here imports antlia. The shapes are plain numpy so the engine can be
exercised on a chain written by hand -- which is how correctness is
established (see CLAUDE.md, "How correctness is established"); a test that
needs a vendor is not a test of the engine.

## Two prices, and they are not interchangeable

Every quote carries both a **close** (the session's last trade) and a **mid**
(the NBBO at the vendor's 17:15 ET stamp). Measured on a year of a large-cap
underlying these
agree to within 1% on 248 of 251 days -- and diverge by 3.5-4.7% on the three
that are earnings dates, where the post-close quote already reflects a
reaction the 16:00 print does not. So:

- **Decide on `close`.** Deciding on `mid` is intraday lookahead.
- **Value at `mid`** (adversely adjusted -- that is the fill model's job).

With one caveat that has no clean fix: an option's `close` is a *last trade*,
and most option rows never trade. Roughly 45% of a full chain has zero volume
on any given day, so `close` is frequently absent or stale on the option side
even though it is reliable on the underlying. A strategy therefore decides off
the **underlying's** close and prices its legs off the 17:15 mid, and that
residual mismatch is a modelling limitation to report, not one to hide.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vela.instrument import Contract, Right

_DAY = "datetime64[D]"


@dataclass(frozen=True, slots=True)
class Quote:
    """One contract's prices on one session."""

    bid: float
    ask: float
    close: float
    volume: int

    @property
    def has_quote(self) -> bool:
        """Whether a usable two-sided quote exists.

        A zero bid is normal on a full chain -- roughly a quarter of the rows
        in a measured large-cap chain had one -- and it is not a quote you can
        sell into.
        """
        return (
            self.bid > 0
            and self.ask > 0
            and self.ask >= self.bid
            and not math.isnan(self.bid)
            and not math.isnan(self.ask)
        )

    @property
    def has_trade(self) -> bool:
        """Whether the session's `close` reflects an actual trade."""
        return self.volume > 0 and self.close > 0 and not math.isnan(self.close)

    @property
    def mid(self) -> float:
        """NBBO midpoint at the 17:15 stamp; NaN without a two-sided quote."""
        return (self.bid + self.ask) / 2 if self.has_quote else math.nan

    @property
    def spread(self) -> float:
        """Absolute bid-ask spread; NaN without a two-sided quote."""
        return self.ask - self.bid if self.has_quote else math.nan


@dataclass(frozen=True, slots=True)
class SpotBar:
    """The underlying's session: OHLCV plus the post-close NBBO."""

    date: dt.date
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: float
    ask: float

    @property
    def quote(self) -> Quote:
        return Quote(bid=self.bid, ask=self.ask, close=self.close, volume=self.volume)

    @property
    def mid(self) -> float:
        return self.quote.mid

    @property
    def divergence(self) -> float:
        """`mid/close - 1`. Large on earnings days -- see the module docstring."""
        if not self.quote.has_quote or self.close <= 0:
            return math.nan
        return self.mid / self.close - 1


@dataclass(slots=True)
class ChainSnapshot:
    """One session's option chain for one underlying, columnar.

    Arrays are parallel and sorted by `(expiration, right, strike)`. They are
    views wherever the producer could manage it, so slicing a window out of a
    year costs nothing.

    Mutable only in its caches: `_index` and `_dates` are built on first use
    by `locate`, which is the lookup the engine does once per open leg per
    day. Screening (`select`) uses vectorised masks and never builds them, so
    a day with no open position never pays for the index.
    """

    date: dt.date
    symbol: str
    expiration: np.ndarray  # datetime64[D]
    strike: np.ndarray  # float64
    right: np.ndarray  # <U1, "C" / "P"
    bid: np.ndarray  # float64
    ask: np.ndarray  # float64
    close: np.ndarray  # float64
    volume: np.ndarray  # int64
    _index: dict[tuple[dt.date, float, str], int] | None = field(default=None, repr=False)
    _dates: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        n = len(self.strike)
        for name in ("expiration", "right", "bid", "ask", "close", "volume"):
            got = len(getattr(self, name))
            if got != n:
                raise ValueError(f"column {name!r} has {got} rows, expected {n}")

    def __len__(self) -> int:
        return int(len(self.strike))

    # -- derived columns ---------------------------------------------------

    @property
    def mid(self) -> np.ndarray:
        """Midpoint per row; NaN where there is no two-sided quote."""
        ok = (self.bid > 0) & (self.ask >= self.bid) & np.isfinite(self.bid) & np.isfinite(self.ask)
        return np.where(ok, (self.bid + self.ask) / 2, np.nan)

    @property
    def dte(self) -> np.ndarray:
        """Calendar days to expiration, date to date (see `Contract.dte`)."""
        return (self.expiration - np.datetime64(self.date, "D")).astype("int64")

    def expirations(self) -> list[dt.date]:
        """Every expiration present today, ascending."""
        uniq = np.unique(self.expiration)
        return [d.astype(object) for d in uniq]

    # -- lookup ------------------------------------------------------------

    def _build_index(self) -> dict[tuple[dt.date, float, str], int]:
        dates = self.expiration.astype(object)
        # Strikes are stored as the vendor sent them, float noise included
        # (antlia never rounds on the way in). Rounding the *key* to four
        # decimals makes a lookup by an identical contract land, and cannot
        # collide: listed strikes are never closer together than $0.50.
        index = {
            (dates[i], round(float(self.strike[i]), 4), str(self.right[i])): i
            for i in range(len(self))
        }
        self._dates = dates
        self._index = index
        return index

    def locate(self, contract: Contract) -> int | None:
        """Row of `contract` today, or None if it is not in the chain.

        O(1). The engine marks every open leg every day, and doing that with a
        boolean scan over the chain -- as the prototype did -- is the single
        hottest wrong thing in a daily options backtest.
        """
        index = self._index if self._index is not None else self._build_index()
        return index.get((contract.expiration, round(contract.strike, 4), contract.right))

    def quote(self, contract: Contract) -> Quote | None:
        """Today's quote for `contract`, or None if it is not in the chain."""
        i = self.locate(contract)
        if i is None:
            return None
        return Quote(
            bid=float(self.bid[i]),
            ask=float(self.ask[i]),
            close=float(self.close[i]),
            volume=int(self.volume[i]),
        )

    def contract_at(self, i: int, symbol: str | None = None) -> Contract:
        """The `Contract` named by row `i`."""
        from vela.instrument import Contract as _Contract

        return _Contract(
            symbol=symbol or self.symbol,
            expiration=self.expiration[i].astype(object),
            strike=float(self.strike[i]),
            right=str(self.right[i]),  # type: ignore[arg-type]
        )

    # -- screening ---------------------------------------------------------

    def select(
        self,
        *,
        expiration: dt.date | None = None,
        right: Right | None = None,
        strikes: tuple[float, float] | None = None,
        dte: tuple[int, int] | None = None,
        quoted: bool = False,
    ) -> ChainSnapshot:
        """The rows matching every filter given, as a new snapshot.

        `quoted=True` keeps only rows with a two-sided quote. That belongs on
        **entry screening** and must never be applied to marking an open
        position: a leg that goes unquoted for a day would silently disappear
        from the book. Two filters, not one `strict` flag.
        """
        keep = np.ones(len(self), dtype=bool)
        if expiration is not None:
            keep &= self.expiration == np.datetime64(expiration, "D")
        if right is not None:
            keep &= self.right == right
        if strikes is not None:
            lo, hi = strikes
            keep &= (self.strike >= lo) & (self.strike <= hi)
        if dte is not None:
            lo_d, hi_d = dte
            days = self.dte
            keep &= (days >= lo_d) & (days <= hi_d)
        if quoted:
            keep &= np.isfinite(self.mid)
        return self._mask(keep)

    def _mask(self, keep: np.ndarray) -> ChainSnapshot:
        return ChainSnapshot(
            date=self.date,
            symbol=self.symbol,
            expiration=self.expiration[keep],
            strike=self.strike[keep],
            right=self.right[keep],
            bid=self.bid[keep],
            ask=self.ask[keep],
            close=self.close[keep],
            volume=self.volume[keep],
        )

    def __repr__(self) -> str:
        return f"<ChainSnapshot {self.symbol} {self.date} rows={len(self)}>"
