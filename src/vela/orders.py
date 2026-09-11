"""What a strategy asks for, and what the engine decides.

The split here is the one that makes sizing a replaceable seam: a strategy
expresses the **shape** of a structure -- which contracts, in what ratio -- and
never how many. The engine prices that unit, asks the `MarginModel` what it
ties up and the `SizingModel` how many to take, and only then commits.

The prototype mixed the two: its signals carried `contracts`, so a strategy
that wanted to size differently had to be edited, and the engine's sizing
could only scale a number the strategy had already chosen. Ratios remove that.
"""

from __future__ import annotations

from dataclasses import dataclass

from vela.instrument import Contract


@dataclass(frozen=True, slots=True)
class LegOrder:
    """One leg of a structure, as a ratio rather than a quantity.

    `ratio` is signed -- positive long, negative short -- and is per **one
    unit** of the structure. A bull put spread is `(-1 short put, +1 long
    put)`; a 1x2 ratio spread is `(-2, +1)`. The engine multiplies by the
    number of units it decides to trade.
    """

    contract: Contract
    ratio: int

    def __post_init__(self) -> None:
        if self.ratio == 0:
            raise ValueError(f"leg {self.contract} has ratio 0; drop it instead")


@dataclass(frozen=True, slots=True)
class Open:
    """Open a new position, all legs filled together or not at all.

    `max_units` caps what the sizing model may take -- a strategy's way of
    saying "no more than this", never "exactly this".
    """

    legs: tuple[LegOrder, ...]
    tag: str = ""
    max_units: int | None = None

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("an Open needs at least one leg")
        seen = {leg.contract for leg in self.legs}
        if len(seen) != len(self.legs):
            raise ValueError("an Open names the same contract twice; combine the ratios")


@dataclass(frozen=True, slots=True)
class Close:
    """Close an open position in full.

    Partial closes and rolls are deliberately absent: nothing in the current
    scope needs them, and inventing the semantics before a strategy demands
    them is how a book model acquires a shape nobody asked for. When they
    arrive they arrive as a field here, not as a new order type.
    """

    position_id: int
    reason: str = "exit"


type Order = Open | Close
