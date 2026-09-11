"""How many units of a structure to take.

Separate from `MarginModel` on purpose. Margin answers what one unit ties up;
sizing answers how many the account should hold given that, the cash on hand,
and whatever the strategy capped it at. For a defined-risk structure the two
numbers coincide -- max loss *is* the requirement -- which is convenient and
not a reason to merge them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class SizingModel(ABC):
    @abstractmethod
    def units(
        self,
        *,
        unit_cash: float,
        unit_margin: float,
        available_cash: float,
        max_units: int | None,
    ) -> int:
        """Units to trade: 0 declines the trade entirely.

        `unit_cash` is signed -- positive for a net credit, negative for a net
        debit. `unit_margin` is what one unit ties up. Returning 0 is a normal
        answer, not an error: an account that cannot afford a structure does
        not trade it.
        """


@dataclass(frozen=True, slots=True)
class FixedUnits(SizingModel):
    """Always the same number of units, cash permitting."""

    n: int = 1

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("FixedUnits needs at least one unit")

    def units(
        self,
        *,
        unit_cash: float,
        unit_margin: float,
        available_cash: float,
        max_units: int | None,
    ) -> int:
        wanted = self.n if max_units is None else min(self.n, max_units)
        return max(0, min(wanted, _affordable(unit_cash, unit_margin, available_cash, wanted)))


def _affordable(unit_cash: float, unit_margin: float, available_cash: float, wanted: int) -> int:
    """How many units the cash on hand covers.

    Cost per unit is the margin it ties up plus any net debit paid for it; a
    net credit reduces the cost but never below the margin, because the credit
    is not spendable while the position is open.
    """
    per_unit = unit_margin + max(-unit_cash, 0.0)
    if per_unit <= 0:
        return wanted
    return int(available_cash // per_unit)


@dataclass(frozen=True, slots=True)
class FixedRisk(SizingModel):
    """Size every entry to roughly the same dollars of capital at risk.

    Raw dollar P&L conflates edge with size: one contract of a $5-wide spread
    and one of a $50-wide spread are not comparable bets, and summing them
    measures strike-width luck as much as signal. Sizing to a constant risk
    budget makes the total a statement about the strategy.

    `budget` is in dollars, matched against whatever the margin model says one
    unit ties up -- which for a defined-risk structure is its max loss, so this
    is literally "risk this much per trade".
    """

    budget: float

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("FixedRisk needs a positive dollar budget")

    def units(
        self,
        *,
        unit_cash: float,
        unit_margin: float,
        available_cash: float,
        max_units: int | None,
    ) -> int:
        if unit_margin <= 0:
            # Nothing to size against. Declining is the honest answer: a
            # structure the margin model says risks nothing is either an
            # arbitrage or a modelling error, and neither should be sized by
            # a rule that divides by risk.
            return 0
        wanted = int(self.budget // unit_margin)
        if max_units is not None:
            wanted = min(wanted, max_units)
        return max(0, min(wanted, _affordable(unit_cash, unit_margin, available_cash, wanted)))
