"""Expiration.

One rule here matters more than the rest: **settlement uses the expiration
date's own spot**, never the price on whatever session the engine happened to
notice the position had expired. Those differ across a weekend or a holiday,
and the difference is silent -- the position settles at a price that never
existed for it. The prototype had that bug.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vela.data.market import MarketView


def settlement_spot(market: MarketView, expiration: dt.date) -> float:
    """The underlying's close on the last session at or before `expiration`.

    Not "the first session after": an expiration that falls on a holiday or a
    Saturday settles against the last price that actually traded before it,
    which is what the exchange does too.
    """
    prior = [d for d in market.sessions if d <= expiration]
    if not prior:
        raise LookupError(f"no session at or before {expiration} to settle against")
    return market.bar(prior[-1]).close
