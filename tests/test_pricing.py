"""Black-Scholes: the inversion, the greeks, and what fails to solve."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from tests.conftest import chain
from vela.pricing.black_scholes import BlackScholes
from vela.pricing.enrich import YEAR_DAYS, Enricher
from vela.pricing.series import atm_iv

BS = BlackScholes()
S, K, T, R, Q = 100.0, 95.0, 0.25, 0.04, 0.005
EXPIRY = dt.date(2026, 2, 20)
DAY = dt.date(2026, 1, 2)


def solve(price: float, *, strike: float = K, is_call: bool = True, dte_years: float = T):
    return BS.greeks_batch(
        price=np.array([price]),
        spot=np.array([S]),
        strike=np.array([strike]),
        dte_years=np.array([dte_years]),
        rate=R,
        is_call=np.array([is_call]),
        dividend_yield=Q,
    )


@pytest.mark.parametrize("right", ["C", "P"])
@pytest.mark.parametrize("vol", [0.08, 0.28, 0.95])
def test_implied_vol_inverts_the_price_exactly(right: str, vol: float) -> None:
    price = BS.price(S, K, T, R, vol, right, Q)
    got = solve(price, is_call=right == "C").iv[0]
    assert got == pytest.approx(vol, abs=1e-8)


def test_put_call_parity_holds(right: str = "C") -> None:
    call = BS.price(S, K, T, R, 0.28, "C", Q)
    put = BS.price(S, K, T, R, 0.28, "P", Q)
    assert call - put == pytest.approx(S * math.exp(-Q * T) - K * math.exp(-R * T), abs=1e-10)


def test_a_deep_wing_still_solves_where_newton_stalls() -> None:
    """Vega collapses far from the money; the bisection fallback covers it."""
    far = 40.0
    price = BS.price(S, far, T, R, 0.9, "P", Q)
    assert price > 0
    assert solve(price, strike=far, is_call=False).iv[0] == pytest.approx(0.9, abs=1e-5)


def test_greek_signs_are_what_they_should_be() -> None:
    call = solve(BS.price(S, K, T, R, 0.28, "C", Q))
    put = solve(BS.price(S, K, T, R, 0.28, "P", Q), is_call=False)
    assert 0 < call.delta[0] < 1
    assert -1 < put.delta[0] < 0
    assert call.gamma[0] > 0 and put.gamma[0] > 0
    assert call.vega[0] > 0 and put.vega[0] > 0
    assert call.theta[0] < 0  # a long option decays
    assert call.rho[0] > 0 and put.rho[0] < 0
    # gamma and vega are strike-and-tenor properties, identical across rights
    assert call.gamma[0] == pytest.approx(put.gamma[0])
    assert call.vega[0] == pytest.approx(put.vega[0])


def test_unsolvable_rows_are_nan_across_every_field() -> None:
    """Not zero. A zero delta and a missing delta are different claims."""
    for bad in (0.0, math.nan, -1.0):
        g = solve(bad)
        assert all(
            math.isnan(float(arr[0])) for arr in (g.iv, g.delta, g.gamma, g.vega, g.theta, g.rho)
        )


def test_an_expired_contract_does_not_solve() -> None:
    g = solve(5.0, dte_years=0.0)
    assert math.isnan(float(g.iv[0]))


def test_solved_mask_reports_what_worked() -> None:
    prices = np.array([BS.price(S, K, T, R, 0.28, "C", Q), 0.0, 4.0])
    g = BS.greeks_batch(
        price=prices,
        spot=np.full(3, S),
        strike=np.full(3, K),
        dte_years=np.full(3, T),
        rate=R,
        is_call=np.ones(3, dtype=bool),
        dividend_yield=Q,
    )
    assert g.solved.tolist() == [True, False, True]


# -- enrichment and the series -------------------------------------------------


def priced_chain(day: dt.date, spot: float, vol: float = 0.25):
    """A chain quoted at Black-Scholes value, so IV must come back as `vol`."""
    rows = []
    for k in (90.0, 95.0, 100.0, 105.0, 110.0):
        years = (EXPIRY - day).days / YEAR_DAYS
        for right in ("P", "C"):
            mid = BS.price(spot, k, years, 0.04, vol, right, 0.0)
            rows.append((k, right, mid - 0.05, mid + 0.05))
    return chain(day, {EXPIRY: rows})


def test_enricher_recovers_the_volatility_it_was_quoted_at() -> None:
    snap = priced_chain(DAY, 100.0, vol=0.25)
    greeks = Enricher(rate=0.04, dividend_yield=0.0).greeks(snap, 100.0)
    assert greeks.solved.all()
    assert np.allclose(greeks.iv, 0.25, atol=2e-3)  # the +/- 0.05 quote width


def test_enricher_handles_an_empty_chain() -> None:
    empty = chain(DAY, {EXPIRY: []})
    assert len(Enricher().greeks(empty, 100.0)) == 0


def test_atm_iv_picks_the_strike_nearest_spot() -> None:
    snap = priced_chain(DAY, 100.0, vol=0.25)
    greeks = Enricher(rate=0.04, dividend_yield=0.0).greeks(snap, 100.0)
    value = atm_iv(snap, greeks, 100.0, target_dte=(EXPIRY - DAY).days)
    assert value == pytest.approx(0.25, abs=2e-3)


def test_atm_iv_is_nan_when_nothing_is_near_the_money() -> None:
    snap = priced_chain(DAY, 100.0)
    greeks = Enricher(rate=0.04, dividend_yield=0.0).greeks(snap, 100.0)
    # spot 102.5 sits between the 100 and 105 strikes; a 0.1% band spans neither
    assert math.isnan(atm_iv(snap, greeks, 102.5, band=0.001))
    assert math.isnan(
        atm_iv(chain(DAY, {EXPIRY: []}), Enricher().greeks(chain(DAY, {EXPIRY: []}), 100.0), 100.0)
    )


def test_no_row_ever_returns_the_seed_unsolved() -> None:
    """The solver's worst failure mode, and the one that looks like success.

    A row Newton abandons -- vega collapsed, or its step left the bracket --
    must go to bisection. Dropping it instead leaves the seed volatility in
    place, and a seed returned as an answer is indistinguishable from a solve
    unless something checks. This checks.
    """
    strikes = np.array([20.0, 35.0, 50.0, 70.0, 95.0, 130.0, 180.0, 260.0])
    vols = np.array([1.20, 0.95, 0.80, 0.55, 0.28, 0.45, 0.70, 1.10])
    prices = np.array([BS.price(S, k, T, R, v, "P", Q) for k, v in zip(strikes, vols, strict=True)])
    solved = BS.greeks_batch(
        price=prices,
        spot=np.full(len(strikes), S),
        strike=strikes,
        dte_years=np.full(len(strikes), T),
        rate=R,
        is_call=np.zeros(len(strikes), dtype=bool),
        dividend_yield=Q,
    )
    unsolvable = prices <= 0
    assert np.allclose(solved.iv[~unsolvable], vols[~unsolvable], atol=1e-5)
