"""Everything a strategy is allowed to see.

The other half of the lookahead barrier. `market` is a `PinnedMarket`, fixed
to today, and `book` is a `BookView`, which has no way to move money. A
strategy that wanted to cheat would have to be handed something else -- and
the engine never hands it anything else.

`chain` and `spot` are **fields rather than lookups** so the engine can
present a hypothetical: the intraday stop pass re-asks the strategy with the
legs repriced at the session's low. Deriving them from `market` would have
made that a wrapper type or a mutation; holding them makes it an argument.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from vela.book.book import BookView
from vela.book.position import Position
from vela.data.market import PinnedMarket
from vela.data.snapshot import ChainSnapshot, SpotBar
from vela.execution.fill import MID
from vela.pricing.base import Greeks


@dataclass(frozen=True, slots=True)
class Context:
    day: dt.date
    market: PinnedMarket
    book: BookView
    #: Today's chain. Decide on `spot.close` -- the 16:00 print. `spot.mid` is
    #: the 17:15 post-close NBBO and, on an earnings day, already knows the
    #: answer.
    chain: ChainSnapshot
    spot: SpotBar
    #: True on a hypothetical re-ask, where the legs have been repriced at the
    #: session's low. **Only a stop answer will be acted on**, so a strategy
    #: must not open here -- an entry at the day's worst print is a fill
    #: nobody offered.
    stressed: bool = False

    @classmethod
    def observed(cls, day: dt.date, market: PinnedMarket, book: BookView) -> Context:
        """The normal case: what actually printed."""
        return cls(day=day, market=market, book=book, chain=market.chain(), spot=market.spot())

    @property
    def symbol(self) -> str:
        return self.market.symbol

    @property
    def greeks(self) -> Greeks:
        """Solved greeks for the **observed** chain, aligned row for row.

        Solved from the 17:15 quotes, so this is a post-close view. Use it to
        describe a position; use `market.greeks_on(a past session)` to make a
        decision, or the signal is reading the day it is trading. On a
        stressed context these still describe the observed chain, not the
        repriced one.
        """
        return self.market.greeks()

    def unrealised_at_mid(self, position: Position) -> float:
        """A position's P&L if closed at the mid, in dollars.

        **Valued at the mid, deliberately.** The engine will cross the spread
        when it actually transacts; a strategy that marked its own positions
        at the adverse side would be optimising against the cost model rather
        than against the market, and would hold winners too long by exactly
        half a spread.

        It is *not* gross of everything: `entry_cash` is net of the commission
        already paid, so that cost is inside this number. That is accounting
        truth rather than a leak -- a trader's P&L on a position does include
        what they paid to open it. What the strategy never sees is the spread
        it has not crossed yet.
        """
        return position.unrealised(self.chain, self.spot.close, MID)
