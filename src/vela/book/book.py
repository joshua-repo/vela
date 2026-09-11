"""The book: cash, and every position sharing it.

Concurrent positions on one cash pool from the first line, rather than the
prototype's single slot. That is not a feature so much as the removal of a
constraint: one position at a time forbids laddered entries, forbids holding
two expirations, and makes "how much capital is this strategy actually using"
an unanswerable question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vela.book.position import Position
from vela.instrument import MULTIPLIER

if TYPE_CHECKING:
    from vela.data.snapshot import ChainSnapshot
    from vela.execution.fill import FillModel


@dataclass(frozen=True, slots=True)
class BookView:
    """What a strategy may see: cash, and the positions, read-only.

    A separate type rather than the `Book` itself, so that a strategy holding
    onto it cannot open or close anything behind the engine's back. The engine
    is the only thing that moves money.
    """

    cash: float
    positions: tuple[Position, ...]

    def __len__(self) -> int:
        return len(self.positions)

    def by_tag(self, tag: str) -> tuple[Position, ...]:
        """Open positions carrying `tag` -- how a strategy finds its own."""
        return tuple(p for p in self.positions if p.tag == tag)


@dataclass(slots=True)
class Book:
    cash: float
    multiplier: int = MULTIPLIER
    _positions: dict[int, Position] = field(default_factory=dict)
    _next_id: int = 1

    @property
    def positions(self) -> tuple[Position, ...]:
        return tuple(self._positions.values())

    def next_id(self) -> int:
        """Reserve the next position id. Ids never repeat within a run."""
        assigned = self._next_id
        self._next_id += 1
        return assigned

    def add(self, position: Position) -> None:
        if position.id in self._positions:
            raise ValueError(f"position id {position.id} is already open")
        self._positions[position.id] = position
        self.cash += position.entry_cash

    def get(self, position_id: int) -> Position:
        try:
            return self._positions[position_id]
        except KeyError:
            raise LookupError(f"no open position {position_id}") from None

    def remove(self, position_id: int, exit_cash: float) -> Position:
        """Close `position_id`, crediting `exit_cash` (signed) to the book."""
        position = self.get(position_id)
        del self._positions[position_id]
        self.cash += exit_cash
        return position

    def market_value(self, chain: ChainSnapshot, spot: float, fill: FillModel) -> float:
        return sum(
            p.market_value(chain, spot, fill, self.multiplier) for p in self._positions.values()
        )

    def equity(self, chain: ChainSnapshot, spot: float, fill: FillModel) -> float:
        """Cash plus what every open position would realise if closed now."""
        return self.cash + self.market_value(chain, spot, fill)

    def unrealised(self, chain: ChainSnapshot, spot: float, fill: FillModel) -> float:
        return sum(
            p.unrealised(chain, spot, fill, self.multiplier) for p in self._positions.values()
        )

    def view(self) -> BookView:
        return BookView(cash=self.cash, positions=self.positions)

    def __len__(self) -> int:
        return len(self._positions)
