"""Hand-built market data, and the local configuration for the few tests
that read a real warmed store.

Every engine test runs on chains written here rather than on a vendor's, so a
failure means the engine is wrong and never that a cache was cold.

A handful of tests do read the real store, to prove the antlia adapter works
end to end. **Which symbol and window they use is local configuration, never
committed**: real research targets stay out of the repository, the same rule
`.vscode/launch.json` follows. Set `VELA_TEST_SYMBOL`, `VELA_TEST_START` and
`VELA_TEST_END` in the environment or in a gitignored `.env`; without them
those tests skip, which is also the right behaviour on a machine with no
cache at all.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from vela.data.market import MarketView
from vela.data.snapshot import ChainSnapshot, SpotBar
from vela.engine.context import Context
from vela.orders import Order
from vela.pricing.enrich import Enricher
from vela.strategy.base import Strategy


def chain(
    day: dt.date,
    expirations: dict[dt.date, list[tuple[float, str, float, float]]],
    symbol: str = "TEST",
) -> ChainSnapshot:
    """A snapshot from `{expiration: [(strike, right, bid, ask), ...]}`."""
    exp, strike, right, bid, ask = [], [], [], [], []
    for expiry, rows in expirations.items():
        for k, r, b, a in rows:
            exp.append(np.datetime64(expiry, "D"))
            strike.append(k)
            right.append(r)
            bid.append(b)
            ask.append(a)
    return ChainSnapshot(
        date=day,
        symbol=symbol,
        expiration=np.asarray(exp, dtype="datetime64[D]"),
        strike=np.asarray(strike, dtype="float64"),
        right=np.asarray(right, dtype="<U1"),
        bid=np.asarray(bid, dtype="float64"),
        ask=np.asarray(ask, dtype="float64"),
        close=np.asarray([(b + a) / 2 for b, a in zip(bid, ask, strict=True)], dtype="float64"),
        volume=np.ones(len(strike), dtype="int64"),
    )


def bar(
    day: dt.date, close: float, *, bid: float | None = None, ask: float | None = None
) -> SpotBar:
    return SpotBar(
        date=day,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=1_000_000,
        bid=bid if bid is not None else close - 0.01,
        ask=ask if ask is not None else close + 0.01,
    )


def _local_env() -> dict[str, str]:
    """Values from a gitignored `.env`, so `uv run pytest` needs no exports.

    Deliberately not python-dotenv: this reads three optional strings for the
    test suite and is not worth a dependency.
    """
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return {}
    found = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            found[key.strip()] = value.strip().strip("\"'")
    return found


def cached_window() -> tuple[str, str, str, int]:
    """The locally configured symbol and window, or skip.

    Skipping is correct rather than a compromise: a checkout with no warmed
    store cannot run these, and neither can one whose owner has not said which
    symbol is theirs.
    """
    env = {**_local_env(), **os.environ}
    symbol = env.get("VELA_TEST_SYMBOL", "")
    start = env.get("VELA_TEST_START", "")
    end = env.get("VELA_TEST_END", "")
    if not (symbol and start and end):
        pytest.skip(
            "set VELA_TEST_SYMBOL / VELA_TEST_START / VELA_TEST_END "
            "(environment or a local .env) to run against a warmed store"
        )
    return symbol, start, end, int(env.get("VELA_TEST_MAX_DTE", "45"))


@pytest.fixture
def simple_chain() -> ChainSnapshot:
    """One expiry, puts and calls at 90/95/100/105/110, spot 100.

    Quotes are intrinsic + 1.00 bid / + 1.40 ask, so every mid is intrinsic
    plus 1.20 of extrinsic and a hand calculation is one line.
    """
    day = dt.date(2026, 1, 2)
    expiry = dt.date(2026, 2, 20)
    rows = []
    for k in (90.0, 95.0, 100.0, 105.0, 110.0):
        put, call = max(k - 100, 0), max(100 - k, 0)
        rows.append((k, "P", put + 1.0, put + 1.4))
        rows.append((k, "C", call + 1.0, call + 1.4))
    return chain(day, {expiry: rows})


def market(
    sessions: dict[dt.date, float],
    chains: dict[dt.date, ChainSnapshot] | None = None,
    symbol: str = "TEST",
    enrich: Enricher | None = None,
) -> MarketView:
    """A MarketView from `{date: spot close}`, built without touching antlia."""
    days = sorted(sessions)
    return MarketView(
        symbol=symbol,
        sessions=days,
        _bars={d: bar(d, sessions[d]) for d in days},
        _chains=dict(chains or {}),
        _enricher=enrich,
    )


class Scripted(Strategy):
    """A strategy that defers to a callable -- the whole point is that a test
    writes the decision inline instead of subclassing for each scenario."""

    def __init__(self, decide: Callable[[Context], list[Order]], name: str = "Scripted") -> None:
        self._decide = decide
        self.name = name

    def on_day(self, ctx: Context) -> list[Order]:
        return self._decide(ctx)
