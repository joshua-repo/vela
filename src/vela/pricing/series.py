"""Turning a chain of implied volatilities into one number per session.

An at-the-money volatility series is the input every volatility signal starts
from, and building it badly is the standard way to manufacture a spurious
cycle.

Two choices matter here:

- **The expiration nearest a fixed target DTE**, not simply the nearest
  expiration. Taking the front month makes the sampled maturity saw-tooth as
  contracts roll, and a wavelet or spectral read of that series then finds a
  monthly cycle that is an artefact of the sampling, not of the market. The
  prototype hit exactly this.
- **The average of the call and the put at the same strike.** Put-call parity
  says they should agree; averaging cancels the small disagreement that
  quoting and the discount-rate assumption introduce, and it makes the series
  insensitive to which side happens to be better quoted that day.

This is still a *sampled* at-the-money volatility, not a constant-maturity
one: the maturity wanders within the tolerance band as expirations roll.
Interpolating in total variance to a fixed tenor is the proper fix and belongs
with a `VolSurface`, which does not exist yet.
"""

from __future__ import annotations

import math

import numpy as np

from vela.data.snapshot import ChainSnapshot
from vela.pricing.base import Greeks


def atm_iv(
    chain: ChainSnapshot,
    greeks: Greeks,
    spot: float,
    *,
    target_dte: int = 37,
    band: float = 0.05,
) -> float:
    """At-the-money implied volatility for one session; NaN when unavailable.

    `band` is the moneyness window a strike must fall inside to count as
    at-the-money, as a fraction of spot.
    """
    if len(chain) == 0 or len(greeks) != len(chain) or not math.isfinite(spot) or spot <= 0:
        return math.nan

    near = np.isfinite(greeks.iv) & (np.abs(chain.strike - spot) <= band * spot)
    if not near.any():
        return math.nan

    dte = chain.dte
    tenor = dte[near][np.argmin(np.abs(dte[near] - target_dte))]
    at_tenor = near & (dte == tenor)

    strikes = chain.strike[at_tenor]
    nearest = strikes[np.argmin(np.abs(strikes - spot))]
    chosen = at_tenor & (chain.strike == nearest)

    return float(np.nanmean(greeks.iv[chosen]))
