"""The only module in Vela that imports antlia.

Everything else works on `ChainSnapshot` and `SpotBar`, which is what lets the
engine be tested on chains written by hand. Swapping the data source, or
adding one, happens here and nowhere else.

## The lookahead barrier

`MarketView` holds the whole window because loading it once is what makes the
run fast. A **strategy never sees a `MarketView`** -- it sees a
`PinnedMarket`, fixed to one session, which raises `FutureData` rather than
quietly answering a question about tomorrow.

That is the difference between a backtest that *was audited* for lookahead and
one that *cannot* have it: the prototype relied on every strategy author
remembering; here the type a strategy is handed makes it impossible.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vela.data.guards import CoverageIncomplete
from vela.data.snapshot import ChainSnapshot, SpotBar

if TYPE_CHECKING:
    from vela.pricing.base import Greeks
    from vela.pricing.enrich import Enricher


@dataclass(frozen=True, slots=True)
class DataVersion:
    """Which ingest of the store a run actually read.

    antlia never edits: a vendor restatement is a **new append**, and a read
    returns the newest version per key unless told otherwise. That makes the
    store honest and a run's result *not reproducible from its own provenance*
    -- same code, same parameters, same git sha, different numbers, and
    nothing to say why.

    Recording the newest `ingested_at` per table closes it. `as_of` is what
    the run asked for: `None` means "whatever was newest at the time", and a
    timestamp means the run was pinned and can be replayed exactly.
    """

    tables: dict[str, dt.datetime | None]
    as_of: dt.datetime | None = None
    #: Whether antlia's coverage check vouched for what was actually read.
    #: **False whenever `as_of` is set**, and not because the check was
    #: skipped out of laziness: coverage is computed from the ledger, which
    #: records the scopes that have *ever* settled and knows nothing about a
    #: pinned read. A pinned read sees a subset, so a `complete` answer about
    #: the live store says nothing about it. Verified at the midpoint of this
    #: store's own warm-up, coverage still reported complete while the read
    #: returned 198,468 of 375,486 rows.
    coverage_verified: bool = False

    @property
    def pinned(self) -> bool:
        return self.as_of is not None

    def __str__(self) -> str:
        held = ", ".join(f"{k}={v.isoformat() if v else 'unknown'}" for k, v in self.tables.items())
        if self.pinned:
            assert self.as_of is not None
            return f"{held} (pinned as_of={self.as_of.isoformat()}; coverage unverifiable)"
        return f"{held}{'' if self.coverage_verified else ' (coverage unverified)'}"


class FutureData(LookupError):
    """A strategy asked for data at or past a session it cannot see yet."""


def _f64(table: Any, name: str) -> np.ndarray:
    """A float column, nulls as NaN."""
    return np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype="float64")


def _i64(table: Any, name: str) -> np.ndarray:
    """An integer column, nulls as 0 -- absent volume means no trade."""
    return np.asarray(table.column(name).fill_null(0).to_numpy(zero_copy_only=False), dtype="int64")


def _days(table: Any, name: str) -> np.ndarray:
    """A date column as `datetime64[D]`."""
    return np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype="datetime64[D]")


@dataclass(frozen=True, slots=True)
class PinnedMarket:
    """What a strategy is allowed to see: one session, and everything before it."""

    day: dt.date
    _view: MarketView

    @property
    def symbol(self) -> str:
        return self._view.symbol

    def spot(self) -> SpotBar:
        """The underlying's bar for today."""
        return self._view.bar(self.day)

    def chain(self) -> ChainSnapshot:
        """Today's option chain."""
        return self._view.chain(self.day)

    def greeks(self) -> Greeks:
        """Today's solved implied volatility and greeks, aligned to the chain."""
        return self._view.greeks(self.day)

    def chain_on(self, day: dt.date) -> ChainSnapshot:
        """A past session's chain. Raises `FutureData` past the pinned day."""
        if day > self.day:
            raise FutureData(f"pinned at {self.day}; {day} has not happened yet in this backtest")
        return self._view.chain(day)

    def greeks_on(self, day: dt.date) -> Greeks:
        """A past session's greeks. Raises `FutureData` past the pinned day.

        Signals want this one: IV is solved from the 17:15 quotes, so today's
        is a post-close quantity and reading it today is intraday lookahead.
        """
        if day > self.day:
            raise FutureData(f"pinned at {self.day}; {day} has not happened yet in this backtest")
        return self._view.greeks(day)

    def bar(self, day: dt.date) -> SpotBar:
        """A past session's bar. Raises `FutureData` past the pinned day."""
        if day > self.day:
            raise FutureData(f"pinned at {self.day}; {day} has not happened yet in this backtest")
        return self._view.bar(day)

    def history(self, sessions: int) -> list[SpotBar]:
        """The last `sessions` bars ending today, oldest first."""
        if sessions < 1:
            raise ValueError("sessions must be at least 1")
        past = [d for d in self._view.sessions if d <= self.day]
        return [self._view.bar(d) for d in past[-sessions:]]


@dataclass(frozen=True, slots=True)
class MarketView:
    """One symbol's window, loaded once, sliced per session.

    Build it with `load()`. The engine holds this; a strategy is handed
    `pin(day)` instead.
    """

    symbol: str
    sessions: list[dt.date]
    _bars: dict[dt.date, SpotBar]
    _chains: dict[dt.date, ChainSnapshot]
    version: DataVersion = field(default_factory=lambda: DataVersion(tables={}))
    _enricher: Enricher | None = None
    #: Solved lazily, per session, and kept. Solving the whole window up front
    #: costs seconds a run that never asks for a greek should not pay.
    _greeks: dict[dt.date, Greeks] = field(default_factory=dict, repr=False)

    @classmethod
    def load(
        cls,
        symbol: str,
        start: str | dt.date,
        end: str | dt.date,
        *,
        options: bool = True,
        enrich: Enricher | None = None,
        as_of: dt.datetime | None = None,
        max_dte: int | None = None,
        min_dte: int | None = None,
        store: str | Path | None = None,
        fetch: bool | None = None,
        guard: bool = True,
        verify: bool = True,
    ) -> MarketView:
        """Read `symbol` over the window from antlia.

        **`max_dte` must match the scope the store was filled with.** antlia
        plans coverage from the expiration listing, so a read that asks about
        every expiration ever listed is uncovered even when every expiration
        you actually fetched is present. Filling with `--max-dte 45` and
        reading without it reports the whole window missing -- correctly, and
        confusingly. Passing the same bound makes the two agree.

        `fetch=None` follows antlia's own default (fill gaps from the vendor);
        `fetch=False` makes the load provably offline, which is also what
        `ANTLIA_HISTORY_FETCH=0` does process-wide.

        `guard=True` runs `check_corporate_actions` and refuses to return a
        window that crosses a split. Turning it off is a deliberate act.

        `verify=True` asks antlia whether it can vouch for the window, which
        is a **stronger question than whether the read succeeded**: a read
        raises only when rows are missing, and says nothing about an
        expiration listing too old to know what should be there.

        `as_of` pins the read to the store as it stood at that moment, so a
        result survives a vendor restatement. Leave it `None` for the newest
        data; the version that was read is recorded either way.
        """
        from antlia import history

        if verify:
            # Still run it when pinned: coverage is an upper bound, so a
            # failure now means the pinned read is certainly short too. A pass
            # proves nothing about it, which is what `coverage_verified` says.
            _verify_coverage(symbol, start, end, options=options, max_dte=max_dte, store=store)

        equity = history.equity_eod(symbol, start, end, store=store, fetch=fetch, as_of=as_of)
        bars = _bars_from_arrow(equity)
        sessions = sorted(bars)

        chains: dict[dt.date, ChainSnapshot] = {}
        if options:
            table = history.option_eod(
                symbol,
                start,
                end,
                max_dte=max_dte,
                min_dte=min_dte,
                store=store,
                fetch=fetch,
                as_of=as_of,
            )
            chains = _chains_from_arrow(table, symbol)

        if guard:
            from vela.data.guards import check_corporate_actions

            check_corporate_actions([bars[d] for d in sessions], chains=chains or None)

        return cls(
            symbol=symbol,
            sessions=sessions,
            _bars=bars,
            _chains=chains,
            version=_read_version(
                symbol,
                options=options,
                store=store,
                as_of=as_of,
                verified=verify and as_of is None,
            ),
            _enricher=enrich,
        )

    def bar(self, day: dt.date) -> SpotBar:
        try:
            return self._bars[day]
        except KeyError:
            raise LookupError(f"no {self.symbol} session on {day}") from None

    def chain(self, day: dt.date) -> ChainSnapshot:
        """Today's chain, or an empty snapshot when none was stored.

        Empty rather than missing, because "no contracts today" is a normal
        thing for the engine to mark against, not an error to unwind a run.
        """
        found = self._chains.get(day)
        if found is not None:
            return found
        return _empty_chain(self.symbol, day)

    def greeks(self, day: dt.date) -> Greeks:
        """Implied volatility and greeks for `day`, solved on first request.

        The solve pairs the chain's 17:15 quotes with the underlying's 17:15
        mid, not its 16:00 close -- see `vela.pricing.enrich`. Falling back to
        the close when no quote exists is the lesser error: a missing spot
        makes every IV on that session NaN.
        """
        if self._enricher is None:
            raise RuntimeError(
                "this MarketView has no Enricher; load it with "
                "MarketView.load(..., enrich=Enricher())"
            )
        cached = self._greeks.get(day)
        if cached is None:
            bar = self.bar(day)
            spot = bar.mid if math.isfinite(bar.mid) else bar.close
            cached = self._enricher.greeks(self.chain(day), spot)
            self._greeks[day] = cached
        return cached

    @property
    def enriched(self) -> bool:
        return self._enricher is not None

    @property
    def enricher(self) -> Enricher | None:
        """The assumptions this view solves volatility under, if any.

        Exposed so a consumer that needs to reprice -- the intraday stop pass
        -- uses the *same* model and rate the IVs it is given were solved
        with. Two sets of assumptions in one run is a silent inconsistency.
        """
        return self._enricher

    def sessions_without_chain(self) -> list[dt.date]:
        """Sessions with a spot but no option rows -- a coverage report."""
        return [d for d in self.sessions if d not in self._chains]

    def pin(self, day: dt.date) -> PinnedMarket:
        return PinnedMarket(day=day, _view=self)

    def __repr__(self) -> str:
        span = f"{self.sessions[0]} .. {self.sessions[-1]}" if self.sessions else "empty"
        return (
            f"<MarketView {self.symbol} {span} "
            f"sessions={len(self.sessions)} chains={len(self._chains)}>"
        )


def data_failure(exc: BaseException) -> str | None:
    """Which kind of data failure `exc` is, or None if it is not one.

    Here rather than in a caller because this is the only module that may name
    antlia's types. The kinds are what a driver does next, so they map to
    distinct exit codes:

    - ``"missing"`` -- the store does not hold the window, or cannot vouch
      for it (`NotCovered`, `CoverageIncomplete`). Warm the cache.
    - ``"denied"`` -- the vendor will never serve it (`NotEntitled`). Change
      the window; filling again cannot help.
    - ``"fetch"`` -- a fetch was attempted and failed (`IngestFailed`).
      Transient until proven otherwise; retry.

    Anything else -- including a bug in the read path -- is None, and should
    surface as the crash it is rather than as advice to warm a cache.
    """
    from antlia.history.errors import IngestFailed, NotCovered, NotEntitled

    from vela.data.guards import CoverageIncomplete

    if isinstance(exc, NotCovered | CoverageIncomplete):
        return "missing"
    if isinstance(exc, NotEntitled):
        return "denied"
    if isinstance(exc, IngestFailed):
        return "fetch"
    return None


def _read_version(
    symbol: str,
    *,
    options: bool,
    store: str | Path | None,
    as_of: dt.datetime | None,
    verified: bool,
) -> DataVersion:
    """The newest append per table, at or before `as_of`.

    Best effort: a store that cannot answer is recorded as `None` rather than
    failing the run. An unknown version is worth saying out loud; it is not
    worth refusing to backtest over.
    """
    from antlia import history

    tables = ["equity_eod"] + (["option_eod"] if options else [])
    found: dict[str, dt.datetime | None] = {}
    for table in tables:
        try:
            stamps = [s for s in history.ingests(table, symbol, store=store) if s is not None]
        except Exception:  # noqa: BLE001 -- provenance must not break a run
            found[table] = None
            continue
        if as_of is not None:
            stamps = [s for s in stamps if s <= as_of]
        found[table] = max(stamps) if stamps else None
    return DataVersion(tables=found, as_of=as_of, coverage_verified=verified)


def _verify_coverage(
    symbol: str,
    start: str | dt.date,
    end: str | dt.date,
    *,
    options: bool,
    max_dte: int | None,
    store: str | Path | None,
) -> None:
    """Refuse a window antlia will not vouch for.

    The option scope must match the fill's, for the same reason the read's
    does: coverage is planned from the expiration listing, and asking about
    every expiration ever listed reports a fully-warmed window as empty.
    """
    from antlia import history

    checks: list[tuple[str, dict[str, Any]]] = [("equity_eod", {})]
    if options:
        checks.append(("option_eod", {"max_dte": max_dte} if max_dte is not None else {}))

    for table, extra in checks:
        found = history.coverage(table, symbol, start, end, store=store, **extra)
        if not found.complete:
            raise CoverageIncomplete(
                f"antlia cannot vouch for this window:\n  {found}\n"
                "Warm the cache (antlia-history fill ...), narrow the window, or "
                "pass verify=False to run on it anyway."
            )


def _empty_chain(symbol: str, day: dt.date) -> ChainSnapshot:
    return ChainSnapshot(
        date=day,
        symbol=symbol,
        expiration=np.empty(0, dtype="datetime64[D]"),
        strike=np.empty(0, dtype="float64"),
        right=np.empty(0, dtype="<U1"),
        bid=np.empty(0, dtype="float64"),
        ask=np.empty(0, dtype="float64"),
        close=np.empty(0, dtype="float64"),
        volume=np.empty(0, dtype="int64"),
    )


def _bars_from_arrow(table: Any) -> dict[dt.date, SpotBar]:
    dates = _days(table, "date")
    cols = {n: _f64(table, n) for n in ("open", "high", "low", "close", "bid", "ask")}
    volume = _i64(table, "volume")
    return {
        dates[i].astype(object): SpotBar(
            date=dates[i].astype(object),
            open=float(cols["open"][i]),
            high=float(cols["high"][i]),
            low=float(cols["low"][i]),
            close=float(cols["close"][i]),
            volume=int(volume[i]),
            bid=float(cols["bid"][i]),
            ask=float(cols["ask"][i]),
        )
        for i in range(table.num_rows)
    }


def _chains_from_arrow(table: Any, symbol: str) -> dict[dt.date, ChainSnapshot]:
    """Split one arrow table into per-session snapshots.

    Sorted once by `(date, expiration, right, strike)` so each session is a
    contiguous run, and every snapshot's arrays are **slices** of the sorted
    columns -- no per-day copy, which is the point of the columnar decision.
    """
    if table.num_rows == 0:
        return {}

    date = _days(table, "date")
    expiration = _days(table, "expiration")
    strike = _f64(table, "strike")
    right = np.asarray(table.column("right").to_numpy(zero_copy_only=False), dtype="object").astype(
        "<U1"
    )
    bid, ask, close = (_f64(table, n) for n in ("bid", "ask", "close"))
    volume = _i64(table, "volume")

    order = np.lexsort((strike, right, expiration, date))
    date, expiration, strike, right = (a[order] for a in (date, expiration, strike, right))
    bid, ask, close, volume = (a[order] for a in (bid, ask, close, volume))

    edges = np.flatnonzero(np.diff(date)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(date)]))

    return {
        date[lo].astype(object): ChainSnapshot(
            date=date[lo].astype(object),
            symbol=symbol,
            expiration=expiration[lo:hi],
            strike=strike[lo:hi],
            right=right[lo:hi],
            bid=bid[lo:hi],
            ask=ask[lo:hi],
            close=close[lo:hi],
            volume=volume[lo:hi],
        )
        for lo, hi in zip(starts, stops, strict=True)
    }
