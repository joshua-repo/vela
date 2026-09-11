"""Early assignment: the short leg that goes away before you meant it to.

American options can be exercised against you at any time, and a backtest
that ignores it is optimistic about every short position it holds. The
decision to exercise early is economic: it is worth doing when the option's
**extrinsic value** falls to nothing, because exercising throws that value
away.

## What this models, and what it does not

`ParityAssignment` fires when a short leg's extrinsic value -- its mid less
its intrinsic -- falls to or below a threshold. At the default threshold of
zero that means the leg is quoted at or below parity, where assignment is
close to certain rather than merely possible.

Two real drivers are approximated rather than modelled:

- **Dividends.** A short call is most likely assigned the evening before an
  ex-dividend date, when the dividend exceeds the remaining extrinsic. Antlia
  ships no dividend calendar, so that timing is invisible here; the parity
  rule catches the same situations late rather than early.
- **Interest.** A deep in-the-money short put is exercised early to free the
  cash. The parity rule catches this directly.

**The whole position terminates, not just the assigned leg.** In reality
assignment on one leg of a spread leaves you holding stock alongside the
remaining option, overnight, with gap risk and a financing cost. Vela's book
holds no equity legs, so modelling that faithfully is out of reach today. The
simplification is deliberately the pessimistic one: terminating forecloses a
recovery the real position might have had.

The default is `ParityAssignment`, not `NoAssignment`, for the same reason the
fill model's default is not free -- an omitted cost is an assumption too, and
the flattering one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vela.book.position import Leg, Position
    from vela.data.snapshot import ChainSnapshot


class AssignmentModel(ABC):
    """Which short legs are taken away overnight."""

    @abstractmethod
    def assigned(self, position: Position, chain: ChainSnapshot, spot: float) -> tuple[Leg, ...]:
        """Short legs deemed assigned after today's close. Empty means none."""


class NoAssignment(AssignmentModel):
    """Nobody is ever assigned. European-style, and optimistic for short books."""

    def assigned(self, position: Position, chain: ChainSnapshot, spot: float) -> tuple[Leg, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class ParityAssignment(AssignmentModel):
    """Assign a short leg once its extrinsic value is gone.

    `threshold` is per share. Zero means "at or below parity". Raising it
    assumes assignment earlier, while some time value remains -- a
    more-pessimistic setting, and a reasonable sensitivity axis.

    A leg with no usable quote is left alone: absence of a quote is not
    evidence of parity, and assuming assignment on it would terminate
    positions for a data gap.
    """

    threshold: float = 0.0

    def assigned(self, position: Position, chain: ChainSnapshot, spot: float) -> tuple[Leg, ...]:
        out = []
        for leg in position.legs:
            if leg.quantity >= 0:
                continue
            quote = chain.quote(leg.contract)
            if quote is None or not quote.has_quote:
                continue
            intrinsic = leg.contract.intrinsic(spot)
            if intrinsic <= 0:
                continue
            if quote.mid - intrinsic <= self.threshold:
                out.append(leg)
        return tuple(out)
