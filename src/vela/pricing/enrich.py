"""Solving a chain's implied volatility, with the assumptions written down.

## Which spot the solve uses, and why it is not the decision spot

The engine decides on the **16:00 close**. This solver uses the **17:15 mid**,
and the difference is deliberate.

Option quotes exist only at the vendor's 17:15 stamp. Pairing them with a
16:00 underlying price would attribute the after-hours move in the option to
volatility instead of to spot -- on an earnings date that is a several-percent
spot error, which lands almost entirely in the solved IV. An implied
volatility is a unit conversion of an observed price, and a unit conversion
has to use quantities observed at the same moment.

The consequence has to be respected rather than hidden: **IV solved this way is
a post-close quantity**. A signal built on it must lag by one session, or it is
reading the earnings reaction on the day of the earnings. `IVRank` does exactly
that.

## The rate and the dividend yield are assumptions, not data

antlia serves neither. Both are constants here, recorded in every run's
`summary.json` so a result says what it assumed. They matter far less than the
spot choice above -- a 1% error in the rate moves a 40-day at-the-money IV by
well under a volatility point -- but "small" is not "absent".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vela.data.snapshot import ChainSnapshot
from vela.instrument import CALL
from vela.pricing.base import Greeks, VolModel
from vela.pricing.black_scholes import BlackScholes

#: Calendar days per year. Calendar rather than trading days, to match the
#: date-to-date DTE convention `Contract.dte` uses -- mixing the two is how a
#: solver quietly returns volatilities scaled by root(252/365).
YEAR_DAYS = 365.0


@dataclass(frozen=True, slots=True)
class Enricher:
    """Solves a chain's greeks under one fixed set of assumptions."""

    model: VolModel = field(default_factory=BlackScholes)
    rate: float = 0.04
    dividend_yield: float = 0.0

    def greeks(self, chain: ChainSnapshot, spot: float) -> Greeks:
        """Implied volatility and every greek for one session's chain.

        `spot` should be the underlying at the **same stamp as the quotes** --
        see the module docstring.
        """
        n = len(chain)
        if n == 0:
            empty = np.empty(0, dtype="float64")
            return Greeks(empty, empty, empty, empty, empty, empty)
        return self.model.greeks_batch(
            price=chain.mid,
            spot=np.full(n, float(spot)),
            strike=chain.strike,
            dte_years=chain.dte / YEAR_DAYS,
            rate=self.rate,
            is_call=chain.right == CALL,
            dividend_yield=self.dividend_yield,
        )
