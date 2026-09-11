"""The strategy interface.

**One method.** The prototype split this into `should_enter` and
`should_exit`, which forced an awkward rule -- `should_enter` had to be called
every day even while a position was open, purely so the strategy could keep
its signal history up to date -- and left "may I do both today" ambiguous. A
single `on_day` returning a list of orders has neither problem: the strategy
sees the whole book and says everything it wants to do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vela.engine.context import Context
from vela.orders import Order


class Strategy(ABC):
    """A decision function over one day of market state."""

    name: str = "Strategy"

    @abstractmethod
    def on_day(self, ctx: Context) -> list[Order]:
        """Everything to do today, or an empty list.

        Called on **every** session, including ones where the book is full and
        nothing can be done -- so a strategy that maintains its own signal
        history has a reliable place to update it.
        """

    def __str__(self) -> str:
        return self.name
