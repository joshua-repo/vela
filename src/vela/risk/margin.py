"""How much capital a structure ties up.

An injected seam, and the engine holds no opinion of its own. The prototype
computed max-loss inline on the engine, which meant "use a different margin
rule" was an engine edit rather than a constructor argument -- and it conflated
two genuinely different questions: *how much capital does this tie up* (margin)
and *how many should I take* (sizing). They are answered by different objects
here even though a defined-risk structure happens to give them the same number.

Portfolio margin (OCC TIMS) is out of reach rather than out of scope: the
parameters are not public. Reg-T is a published rule set, and that is what
gets implemented.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vela.book.payoff import terminal_bounds
from vela.instrument import MULTIPLIER

if TYPE_CHECKING:
    from vela.book.position import Leg


class MarginModel(ABC):
    """What holding a set of legs costs in tied-up capital."""

    @abstractmethod
    def requirement(self, legs: Sequence[Leg], spot: float, multiplier: int = MULTIPLIER) -> float:
        """Dollars of capital required to hold `legs`, never negative."""


class UnboundedRisk(ValueError):
    """A structure whose worst case has no finite bound."""


@dataclass(frozen=True, slots=True)
class MaxLossMargin(MarginModel):
    """Capital required is the structure's worst-case terminal loss.

    The expiration payoff of any combination of options is piecewise linear in
    spot, so its minimum lies at a kink -- zero, one of the strikes, or far
    beyond the highest. Evaluating those points is exact, not a sample.

    What this equals, and where it does not:

    - **A vertical spread: exactly Reg-T.** Short a 95 put and long a 90 put
      for a credit `c` loses at most `500 - c`, which is width x multiplier
      less the credit -- the published rule, arrived at from the payoff rather
      than copied from it.
    - **A cash-secured put: the cash-secured amount**, which is *more* than
      Reg-T's naked-put rule (the greater of 20% of spot less the out-of-the-
      money amount, or 10% of the strike). Conservative and defensible, but
      not the same number, so it is named honestly.
    - **A naked short call: refused.** The loss is unbounded above and no
      finite requirement describes it. Raising beats returning a large number
      that silently caps a risk which has no cap.

    `entry_price` is folded in, so the requirement is net of premium already
    exchanged -- which is what "capital tied up" means once a position is on.
    """

    def requirement(self, legs: Sequence[Leg], spot: float, multiplier: int = MULTIPLIER) -> float:
        worst, _ = terminal_bounds(legs, multiplier)
        if worst == -math.inf:
            raise UnboundedRisk(
                "net short calls have no bounded worst case; MaxLossMargin cannot "
                "size this structure. Use a margin model with an explicit naked rule."
            )
        return max(-worst, 0.0)


class NoMargin(MarginModel):
    """Capital is unconstrained.

    A diagnostic, not a default. It answers "what would this strategy make
    with unlimited capital", which is a real question -- but a portfolio
    engine under `NoMargin` will hold as many concurrent positions as the
    strategy asks for, so its return figures describe no attainable account.
    """

    def requirement(self, legs: Sequence[Leg], spot: float, multiplier: int = MULTIPLIER) -> float:
        return 0.0
