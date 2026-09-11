"""Black-Scholes with a continuous dividend yield, vectorised.

The model is wrong in known ways for American options -- it prices no early
exercise -- and that is accepted here for a specific reason: **implied
volatility is a unit conversion of an observed price**, not a forecast.
`iv = BS^-1(quote)` turns a dollar quote into a number that is comparable
across strikes and dates. The anchor is the market's own price; the model is
the ruler.

Where that breaks down is deep in the money, where the American premium is
real and the European inversion attributes it to volatility. Those are exactly
the rows a near-the-money signal does not use, and the reason a binomial model
is on the list rather than in the code.

## Solving

Newton on vega, with a bisection fallback per element that Newton left
unconverged -- vega collapses far from the money and Newton stalls there,
which is where a pure-Newton solver quietly returns nonsense. The bisection
bracket is `1e-4 .. 5.0`; a quote that solves to the floor is unpriceable
rather than very cheap, and reads as such.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from vela.pricing.base import Greeks, VolModel

_TOL = 1e-6
_NEWTON_STEPS = 50
_BISECT_STEPS = 100
_VOL_FLOOR = 1e-4
_VOL_CEILING = 5.0
_SEED_VOL = 0.3


def _pdf(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
    return out


def _cdf(x: np.ndarray) -> np.ndarray:
    """The standard normal CDF. `scipy.special.ndtr` is exact and vectorised;
    an erf approximation good to 1e-7 is not enough under a 1e-6 solver."""
    out: np.ndarray = ndtr(x)
    return out


class BlackScholes(VolModel):
    name = "BlackScholes"

    # -- scalar ------------------------------------------------------------

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
        value = self._price_vec(
            np.array([spot]),
            np.array([strike]),
            np.array([dte_years]),
            np.array([rate]),
            np.array([vol]),
            np.array([right.upper().startswith("C")]),
            dividend_yield,
        )[0]
        return float(value)

    # -- vectorised core ---------------------------------------------------

    @staticmethod
    def _d1_d2(
        s: np.ndarray,
        k: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        vol: np.ndarray,
        q: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        root_t = np.sqrt(t)
        d1 = (np.log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * root_t)
        d2: np.ndarray = d1 - vol * root_t
        return d1, d2

    def _price_vec(
        self,
        s: np.ndarray,
        k: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        vol: np.ndarray,
        is_call: np.ndarray,
        q: float,
    ) -> np.ndarray:
        d1, d2 = self._d1_d2(s, k, t, r, vol, q)
        disc_q, disc_r = np.exp(-q * t), np.exp(-r * t)
        call = s * disc_q * _cdf(d1) - k * disc_r * _cdf(d2)
        put = k * disc_r * _cdf(-d2) - s * disc_q * _cdf(-d1)
        priced: np.ndarray = np.where(is_call, call, put)
        return priced

    def _price_and_vega(
        self,
        s: np.ndarray,
        k: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        vol: np.ndarray,
        is_call: np.ndarray,
        q: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        d1, d2 = self._d1_d2(s, k, t, r, vol, q)
        disc_q, disc_r = np.exp(-q * t), np.exp(-r * t)
        call = s * disc_q * _cdf(d1) - k * disc_r * _cdf(d2)
        put = k * disc_r * _cdf(-d2) - s * disc_q * _cdf(-d1)
        vega = s * disc_q * _pdf(d1) * np.sqrt(t)
        priced: np.ndarray = np.where(is_call, call, put)
        return priced, vega

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
        p = np.asarray(price, dtype="float64")
        s = np.asarray(spot, dtype="float64")
        k = np.asarray(strike, dtype="float64")
        t = np.asarray(dte_years, dtype="float64")
        r = np.full(p.shape, float(rate))
        call = np.asarray(is_call, dtype=bool)
        q = float(dividend_yield)

        vol = self._solve(p, s, k, t, r, call, q)
        return self._greeks_at(vol, s, k, t, r, call, q)

    def _solve(
        self,
        p: np.ndarray,
        s: np.ndarray,
        k: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        call: np.ndarray,
        q: float,
    ) -> np.ndarray:
        """Implied volatility per row; NaN where the quote cannot produce one."""
        usable = (t > 0) & (p > 0) & (s > 0) & (k > 0) & np.isfinite(p)
        vol = np.where(usable, _SEED_VOL, np.nan)
        active = usable.copy()
        # Rows Newton could not carry: vega collapsed, or the step left the
        # bracket. They must be *handed to bisection*, never merely dropped --
        # a dropped row keeps the seed and returns it as though it had solved,
        # which is a wrong answer that looks exactly like a right one.
        stalled = np.zeros(vol.shape, dtype=bool)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for _ in range(_NEWTON_STEPS):
                if not active.any():
                    break
                model, vega = self._price_and_vega(s, k, t, r, vol, call, q)
                active = active & (np.abs(model - p) >= _TOL)
                if not active.any():
                    break

                step = np.where(vega > 1e-12, (model - p) / vega, np.nan)
                nxt = vol - step
                ok = np.isfinite(nxt) & (nxt > _VOL_FLOOR) & (nxt < _VOL_CEILING)

                stalled |= active & ~ok
                active = active & ok
                vol = np.where(active, nxt, vol)

            # Whatever Newton abandoned, plus whatever it never finished.
            fallback = usable & (stalled | active)
            if fallback.any():
                vol[fallback] = self._bisect(
                    p[fallback],
                    s[fallback],
                    k[fallback],
                    t[fallback],
                    r[fallback],
                    call[fallback],
                    q,
                )
        vol[~usable] = np.nan
        return vol

    def _bisect(
        self,
        p: np.ndarray,
        s: np.ndarray,
        k: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        call: np.ndarray,
        q: float,
    ) -> np.ndarray:
        """Bisection on the bracket. Monotone in vol, so it always converges."""
        lo = np.full(p.shape, _VOL_FLOOR)
        hi = np.full(p.shape, _VOL_CEILING)
        for _ in range(_BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            model = self._price_vec(s, k, t, r, mid, call, q)
            too_cheap = model < p
            lo = np.where(too_cheap, mid, lo)
            hi = np.where(too_cheap, hi, mid)
        solved: np.ndarray = 0.5 * (lo + hi)
        return solved

    def _greeks_at(
        self,
        vol: np.ndarray,
        s: np.ndarray,
        k: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        call: np.ndarray,
        q: float,
    ) -> Greeks:
        nan = np.full(vol.shape, np.nan)
        good = np.isfinite(vol) & (vol > _VOL_FLOOR) & (t > 0)
        if not good.any():
            return Greeks(vol, nan, nan.copy(), nan.copy(), nan.copy(), nan.copy())

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            d1, d2 = self._d1_d2(s, k, t, r, vol, q)
            root_t = np.sqrt(t)
            disc_q, disc_r = np.exp(-q * t), np.exp(-r * t)
            pdf_d1 = _pdf(d1)

            delta = np.where(call, disc_q * _cdf(d1), disc_q * (_cdf(d1) - 1.0))
            gamma = disc_q * pdf_d1 / (s * vol * root_t)
            vega = s * disc_q * pdf_d1 * root_t
            decay = -s * disc_q * pdf_d1 * vol / (2 * root_t)
            theta = np.where(
                call,
                decay - r * k * disc_r * _cdf(d2) + q * s * disc_q * _cdf(d1),
                decay + r * k * disc_r * _cdf(-d2) - q * s * disc_q * _cdf(-d1),
            )
            rho = np.where(call, k * t * disc_r * _cdf(d2), -k * t * disc_r * _cdf(-d2))

        return Greeks(
            iv=np.where(good, vol, np.nan),
            delta=np.where(good, delta, np.nan),
            gamma=np.where(good, gamma, np.nan),
            vega=np.where(good, vega, np.nan),
            theta=np.where(good, theta, np.nan),
            rho=np.where(good, rho, np.nan),
        )
