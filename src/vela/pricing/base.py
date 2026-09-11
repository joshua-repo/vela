"""The pricing seam and the conventions it pins.

Conventions are fixed **here**, once, because a greek whose units are
ambiguous is worse than no greek at all:

- `iv` is annualised, as a decimal: 0.25 means 25%.
- `vega` is raw `dPrice/dVol` -- per 1.00 of volatility, not per point.
  Divide by 100 for the trader's convention.
- `theta` is per **year**. Divide by 365 for per-day.
- `rho` is per 1.00 of rate.
- `delta` and `gamma` are per share of the underlying.

Anything unsolvable comes back `NaN` across *every* field of that row, never
zero. A zero delta and a missing delta are different claims, and only one of
them is true of an unpriceable quote.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Greeks:
    """One aligned array per quantity, NaN where the solve failed."""

    iv: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    rho: np.ndarray

    def __len__(self) -> int:
        return int(len(self.iv))

    @property
    def solved(self) -> np.ndarray:
        """Boolean mask of rows an implied volatility was found for."""
        return np.isfinite(self.iv)


class VolModel(ABC):
    """A pricing model. Two methods, and the batch one is the hot path.

    `greeks_batch` is where a faster implementation slots in -- the same
    signature backed by a compiled kernel, with every caller unchanged. That
    is the pattern the prototype validated: a Rust batch solver behind an
    identical interface ran ~100x faster with the Python version retained as
    the reference oracle.
    """

    name: str = "VolModel"

    @abstractmethod
    def price(
        self,
        spot: float,
        strike: float,
        dte_years: float,
        rate: float,
        vol: float,
        right: str,
        dividend_yield: float = 0.0,
    ) -> float:
        """Theoretical price of one contract, per share."""

    @abstractmethod
    def greeks_batch(
        self,
        *,
        price: np.ndarray,
        spot: np.ndarray,
        strike: np.ndarray,
        dte_years: np.ndarray,
        rate: float,
        is_call: np.ndarray,
        dividend_yield: float = 0.0,
    ) -> Greeks:
        """Solve implied volatility and every greek for whole arrays at once."""
