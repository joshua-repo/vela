"""Turning a quote into a volatility, and a volatility into risk.

Deliberately absent until now. A vertical spread needs no pricing model:
strikes are picked by moneyness, exits fire on realised P&L, and open legs are
marked off observed quotes. Pricing arrives for a different reason -- **a
strategy with no signal has nothing worth sweeping**, and every volatility
signal starts with an implied volatility.

`VolModel` is a seam, exactly like `MarginModel` and `SizingModel`.
`greeks_batch` is the hot path and the intended hand-off point for the
separate compute project: arrays in, arrays out, callers unchanged.
"""

from vela.pricing.base import Greeks, VolModel
from vela.pricing.black_scholes import BlackScholes
from vela.pricing.enrich import Enricher
from vela.pricing.series import atm_iv

__all__ = ["BlackScholes", "Enricher", "Greeks", "VolModel", "atm_iv"]
