"""What a leg actually fills at, and what the trade costs.

Strategies decide off quotes and stay cost-unaware; every adverse adjustment
lives here. That separation is what lets the same strategy be run costed and
uncosted without touching it.

**The default is not free.** antlia's own measurement of a large-cap option
chain found
median spreads of 6.3% above $5, 9.4% from $1-5 and 24.7% from $0.10-1, and
the quotes are a 17:15 post-close snapshot rather than a print you could have
traded on. A zero-cost default would make every result optimistic by an amount
of the same order as the edge being measured, so the default crosses half a
half-spread and charges a real commission. `haircut=0.0, commission=0.0`
reproduces an uncosted run exactly, and that is a deliberate diagnostic, not
the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vela.data.snapshot import Quote


@dataclass(frozen=True, slots=True)
class Costs:
    """What execution took out, split by where it went.

    Recorded on every trade rather than reconstructed by a special study,
    because on a defined-risk options strategy the spread is not a detail
    around the edge of the result -- it frequently *is* the result. Measured
    on a costed run of bullish verticals: the crossing cost ran roughly four
    times the commission, and roughly three times the gross edge, so a
    profitable strategy before execution was a losing one after it.

    A result that reports only net P&L hides which of those a reader should
    argue with.
    """

    spread: float = 0.0
    commission: float = 0.0

    @property
    def total(self) -> float:
        return self.spread + self.commission

    def __add__(self, other: Costs) -> Costs:
        return Costs(self.spread + other.spread, self.commission + other.commission)


class Side(Enum):
    BUY = 1
    SELL = -1


BUY = Side.BUY
SELL = Side.SELL


@dataclass(frozen=True, slots=True)
class FillModel:
    """Adverse fill pricing and per-contract commission.

    `haircut` is in **half-spreads**: 0 fills at the mid, 1 fills at the touch
    (bid when selling, ask when buying), 0.5 splits the difference.
    """

    haircut: float = 0.5
    commission: float = 0.65

    def __post_init__(self) -> None:
        if self.haircut < 0:
            raise ValueError("haircut must not be negative -- it is an adverse adjustment")
        if self.commission < 0:
            raise ValueError("commission must not be negative")

    def haircut_on(self, quote: Quote | None) -> float:
        """Per-share adverse adjustment this model applies to `quote`.

        Zero without a two-sided quote: there is no spread to cross, and the
        intrinsic fallback is not a crossing.
        """
        if quote is None or not quote.has_quote or self.haircut <= 0:
            return 0.0
        return self.haircut * quote.spread / 2

    def price(self, quote: Quote | None, side: Side, *, intrinsic: float) -> float:
        """Per-share price for transacting `side`.

        Falls back to `intrinsic` when there is no two-sided quote. That is
        the conservative answer and, more importantly, an honest one: a leg
        nobody is quoting cannot be marked at yesterday's mid as though it
        still traded there. The prototype carried the stale entry price
        forward, which flatters a position exactly when it is in trouble.
        """
        if quote is None or not quote.has_quote:
            return intrinsic
        adjustment = self.haircut_on(quote)
        return quote.mid + adjustment if side is BUY else quote.mid - adjustment

    def commission_on(self, contracts: int) -> float:
        """Commission in dollars for `contracts` contracts, sign-insensitive."""
        return self.commission * abs(contracts)


#: A cost-free fill model. Strategies evaluate their own positions with this:
#: they decide off mid quotes and stay cost-unaware, while the engine fills
#: adversely with the real one. Also the uncosted diagnostic baseline.
MID = FillModel(haircut=0.0, commission=0.0)
