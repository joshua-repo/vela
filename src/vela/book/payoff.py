"""The terminal payoff of a combination of options, and its two bounds.

At expiration the P&L of any set of legs is **piecewise linear in spot**, with
kinks only at the strikes. So its extremes lie at a kink, at zero, or beyond
the highest strike -- evaluating those points is exact, not a sample of a
curve.

Both bounds are wanted, and by different callers, which is why they are
computed together here rather than inside either of them:

- `worst` is what a margin model charges for.
- `best` is what a profit target is a fraction of.

Getting that second one wrong is not a rounding error. A credit spread's best
case *is* the credit -- a $5-wide spread taken for $1.90 can never make more
than $190 while tying up $810 -- so a take-profit expressed as a fraction of
capital at risk can be mathematically unreachable, and a backtest will run
clean and simply never take a profit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vela.instrument import MULTIPLIER

if TYPE_CHECKING:
    from vela.book.position import Leg


def terminal_bounds(legs: Sequence[Leg], multiplier: int = MULTIPLIER) -> tuple[float, float]:
    """`(worst, best)` terminal P&L in dollars, net of the premium exchanged.

    Infinite where the payoff genuinely is: net short calls lose without
    bound, net long calls gain without bound. Returning a large finite number
    instead would let a caller size against a risk that has no size.
    """
    if not legs:
        return 0.0, 0.0

    net_calls = sum(leg.quantity for leg in legs if leg.contract.right == "C")
    strikes = [leg.contract.strike for leg in legs]
    probes = [0.0, *strikes, max(strikes) * 2 + 1]
    values = [_payoff(legs, probe, multiplier) for probe in probes]

    worst = -math.inf if net_calls < 0 else min(values)
    best = math.inf if net_calls > 0 else max(values)
    return worst, best


def _payoff(legs: Sequence[Leg], spot: float, multiplier: int) -> float:
    return sum(
        leg.quantity * (leg.contract.intrinsic(spot) - leg.entry_price) * multiplier for leg in legs
    )
