"""An open position: what is held, at what price, and what it is worth now.

## Everything here is signed dollars

The prototype carried money "per share, with contracts folded in, multiply by
the multiplier for dollars", and every reader had to hold that convention in
their head to check an arithmetic step. Vela uses **dollars, signed from the
account's point of view**, everywhere: a credit received is positive, a debit
paid is negative, and the multiplier is applied once, at the fill boundary.

That choice is what makes the cash invariant checkable in one line:

    equity - initial_cash == realised P&L + unrealised P&L

with `unrealised = entry_cash + market_value` for every open position.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vela.book.payoff import terminal_bounds
from vela.execution.fill import BUY, SELL, Costs, FillModel
from vela.instrument import MULTIPLIER, Contract

if TYPE_CHECKING:
    from vela.data.snapshot import ChainSnapshot


@dataclass(frozen=True, slots=True)
class Leg:
    """One leg as actually filled. `quantity` is signed: long positive."""

    contract: Contract
    quantity: int
    entry_price: float

    def __post_init__(self) -> None:
        if self.quantity == 0:
            raise ValueError(f"leg {self.contract} filled 0 contracts")


@dataclass(frozen=True, slots=True)
class Position:
    """A multi-leg structure, held as one thing.

    Held as one thing because that is how it is risked and closed. Two legs of
    a spread tracked independently can be closed independently, which silently
    turns a defined-risk position into a naked one -- the margin model would
    then be describing a structure the book no longer holds.
    """

    id: int
    legs: tuple[Leg, ...]
    opened_on: dt.date
    entry_spot: float
    entry_cash: float
    units: int = 1
    #: What the margin model said one held structure ties up, at entry. Kept
    #: for reporting return-on-capital; the live constraint is recomputed
    #: daily, so a model whose requirement moves with spot stays correct.
    entry_margin: float = 0.0
    #: What opening cost: spread crossed and commission paid.
    entry_costs: Costs = Costs()
    tag: str = ""

    @property
    def expiration(self) -> dt.date:
        """The earliest leg expiration -- when this structure stops existing."""
        return min(leg.contract.expiration for leg in self.legs)

    @property
    def contracts(self) -> int:
        """Total contracts across every leg, for commission purposes."""
        return sum(abs(leg.quantity) for leg in self.legs)

    @property
    def max_loss(self) -> float:
        """Worst-case terminal loss in dollars, positive. `inf` if unbounded."""
        worst, _ = terminal_bounds(self.legs)
        return -worst

    @property
    def max_profit(self) -> float:
        """Best-case terminal gain in dollars. `inf` if unbounded.

        A credit spread's best case is the credit it was opened for -- which
        is why a profit target must be a fraction of *this*, not of the
        capital the structure ties up.
        """
        _, best = terminal_bounds(self.legs)
        return best

    def market_value(
        self, chain: ChainSnapshot, spot: float, fill: FillModel, multiplier: int = MULTIPLIER
    ) -> float:
        """Signed dollars realised by closing this position right now.

        **Realisable, not the mid.** A short leg is bought back at the ask
        side and a long leg sold at the bid side, through the same fill model
        a real close would use. Marking at the mid overstates every position
        by half the spread, which on a $0.50-wide option quote is most of the
        edge. Commission is not deducted here -- it is charged on the fill
        that actually happens, and a mark is not a fill.
        """
        total = 0.0
        for leg in self.legs:
            side = SELL if leg.quantity > 0 else BUY
            price = fill.price(
                chain.quote(leg.contract), side, intrinsic=leg.contract.intrinsic(spot)
            )
            total += leg.quantity * price * multiplier
        return total

    def close_out(
        self, chain: ChainSnapshot, spot: float, fill: FillModel, multiplier: int = MULTIPLIER
    ) -> tuple[float, Costs]:
        """Close now: the net cash it produces, and what execution took.

        Separate from `market_value` because a mark is not a fill. Marking
        must not accrue costs, or every held day would charge for a trade
        that never happened.
        """
        proceeds = 0.0
        spread = 0.0
        for leg in self.legs:
            quote = chain.quote(leg.contract)
            side = SELL if leg.quantity > 0 else BUY
            price = fill.price(quote, side, intrinsic=leg.contract.intrinsic(spot))
            proceeds += leg.quantity * price * multiplier
            spread += fill.haircut_on(quote) * abs(leg.quantity) * multiplier
        commission = fill.commission_on(self.contracts)
        return proceeds - commission, Costs(spread, commission)

    def settle_out(
        self, spot: float, fill: FillModel, multiplier: int = MULTIPLIER
    ) -> tuple[float, Costs]:
        """Let it expire: net cash, and the exercise fees on in-the-money legs.

        No spread cost -- settlement happens at intrinsic, with nobody to
        cross.
        """
        commission = fill.commission_on(self.exercised_contracts(spot))
        return self.settlement_value(spot, multiplier) - commission, Costs(0.0, commission)

    def assign_out(
        self, chain: ChainSnapshot, spot: float, fill: FillModel, multiplier: int = MULTIPLIER
    ) -> tuple[float, Costs]:
        """Unwind after early assignment: net cash, and what it cost.

        Short legs are exercised against and settle at intrinsic -- there is
        no spread to cross, the holder simply exercises. Long legs are sold at
        market, and that sale does cross one.
        """
        proceeds = 0.0
        spread = 0.0
        for leg in self.legs:
            if leg.quantity < 0:
                proceeds += leg.quantity * leg.contract.intrinsic(spot) * multiplier
                continue
            quote = chain.quote(leg.contract)
            price = fill.price(quote, SELL, intrinsic=leg.contract.intrinsic(spot))
            proceeds += leg.quantity * price * multiplier
            spread += fill.haircut_on(quote) * abs(leg.quantity) * multiplier
        commission = fill.commission_on(self.contracts)
        return proceeds - commission, Costs(spread, commission)

    def settlement_value(self, spot: float, multiplier: int = MULTIPLIER) -> float:
        """Signed dollars from letting every leg expire at `spot`."""
        return sum(leg.quantity * leg.contract.intrinsic(spot) * multiplier for leg in self.legs)

    def exercised_contracts(self, spot: float) -> int:
        """Contracts finishing in the money -- the only ones that cost a fee.

        An option expiring worthless is not exercised and carries no charge.
        The prototype commissioned every leg at expiry, overstating the cost
        of exactly the outcome a credit strategy is aiming for.
        """
        return sum(abs(leg.quantity) for leg in self.legs if leg.contract.intrinsic(spot) > 0)

    def unrealised(
        self, chain: ChainSnapshot, spot: float, fill: FillModel, multiplier: int = MULTIPLIER
    ) -> float:
        """P&L if closed now: what came in at entry plus what closing returns."""
        return self.entry_cash + self.market_value(chain, spot, fill, multiplier)

    def __str__(self) -> str:
        legs = ", ".join(f"{leg.quantity:+d} {leg.contract}" for leg in self.legs)
        return f"#{self.id} [{legs}]"
