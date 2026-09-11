"""The daily loop.

The order of operations inside a session is a design decision, not an
implementation detail:

    1. ask the strategy          -- it sees the book and today's chain
    2. execute closes            -- frees capital before anything is sized
    3. execute opens             -- margin-gated, then sized
    4. settle expiries           -- anything the strategy chose to keep
    5. record the day            -- marked at realisable prices

**Closes before opens** because a portfolio engine on one cash pool otherwise
declines an entry it could afford, purely because the capital freeing up this
session had not been freed yet. That is an artefact of loop order, and it
would look like a strategy result.

**Settlement after the strategy acts** because an option trades all day on its
expiration date. Settling first takes away a decision the strategy really had.

**Settlement at the expiration date's own spot**, resolved through
`settlement_spot`, not at whatever session the engine noticed on. Those differ
across a weekend, and the difference is silent.

**Assignment is checked last**, after everything else has had its turn: you
are assigned on the evening of session D and find out on the morning of D+1,
so a strategy that closed the position during D was never assigned at all.
Checking earlier would take away a decision the trader really had.

**A blacked-out session still calls the strategy.** `blackout` suppresses the
*orders*, not the call: a gate maintaining its own history has to advance on
days it cannot trade, or its range is computed only from the days it happened
to be allowed. The suppressed orders are recorded as declined, so a run says
how often the policy bit.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace

from vela.book.book import Book
from vela.book.position import Leg, Position
from vela.data.guards import DIVERGENCE, quote_divergence
from vela.data.market import MarketView
from vela.data.snapshot import ChainSnapshot
from vela.engine.context import Context
from vela.engine.intraday import eligible, leg_ivs, stress_chain
from vela.engine.results import DataQuality, Declined, EquityPoint, Result, Trade
from vela.execution.assignment import AssignmentModel, ParityAssignment
from vela.execution.fill import BUY, SELL, Costs, FillModel
from vela.execution.settle import settlement_spot
from vela.instrument import MULTIPLIER, Contract
from vela.orders import Close, Open
from vela.risk.margin import MarginModel, NoMargin
from vela.risk.sizing import FixedUnits, SizingModel
from vela.strategy.base import Strategy


@dataclass(slots=True)
class BacktestEngine:
    """Runs one strategy over one symbol's window.

    Every replaceable decision is a constructor argument: how a leg fills,
    what capital it ties up, how many units to take. The engine itself decides
    only *when* things happen.
    """

    fill: FillModel = field(default_factory=FillModel)
    assignment: AssignmentModel = field(default_factory=ParityAssignment)
    #: Refuse to act on sessions where the 17:15 NBBO mid is further than this
    #: fraction from the 16:00 close. Those are earnings dates, where the
    #: option quotes a strategy selects and prices from already contain a
    #: reaction the decision price does not -- lookahead that EOD data cannot
    #: repair, only avoid. None trades through them and reports the count.
    blackout: float | None = None
    #: Re-ask the strategy with the open legs repriced at the session's low,
    #: and act only on a stop. This is the pessimistic half of the envelope --
    #: see `vela.engine.intraday`. Requires an enriched `MarketView`.
    intraday_stop: bool = False
    margin: MarginModel = field(default_factory=NoMargin)
    sizing: SizingModel = field(default_factory=FixedUnits)
    initial_cash: float = 100_000.0
    multiplier: int = MULTIPLIER

    def run(
        self,
        market: MarketView,
        strategy: Strategy,
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> Result:
        sessions = [
            d
            for d in market.sessions
            if (start is None or d >= start) and (end is None or d <= end)
        ]
        if not sessions:
            raise ValueError("no sessions in the requested window")

        book = Book(cash=self.initial_cash, multiplier=self.multiplier)
        result = Result(initial_cash=self.initial_cash)
        realised = 0.0
        no_chain = marked = unquoted = 0
        prior_iv: dict[Contract, float] = {}
        bars = [market.bar(d) for d in sessions]
        divergent = {f.date for f in quote_divergence(bars, threshold=DIVERGENCE)}
        blacked_out = (
            {f.date for f in quote_divergence(bars, threshold=self.blackout)}
            if self.blackout is not None
            else set()
        )

        for day in sessions:
            chain = market.chain(day)
            spot = market.bar(day).close
            if not len(chain):
                no_chain += 1

            # Called even when blacked out: a gate's history must not have
            # holes on the days its own policy shut it.
            orders = strategy.on_day(Context.observed(day, market.pin(day), book.view()))
            if day in blacked_out:
                result.declined.extend(
                    Declined(day, _tag_of(order), "blackout: quotes diverge from the close")
                    for order in orders
                )
                orders = []

            for order in orders:
                if isinstance(order, Close):
                    outcome = self._close(book, order, chain, spot, day)
                    if isinstance(outcome, Trade):
                        result.trades.append(outcome)
                        realised += outcome.pnl
                    else:
                        result.declined.append(outcome)

            for order in orders:
                if isinstance(order, Open):
                    declined = self._open(book, order, chain, spot, day)
                    if declined is not None:
                        result.declined.append(declined)

            if self.intraday_stop:
                closed, prior_iv = self._intraday(
                    book, market, strategy, day, chain, spot, prior_iv
                )
                for trade in closed:
                    result.trades.append(trade)
                    realised += trade.pnl

            for position in [p for p in book.positions if p.expiration <= day]:
                trade = self._settle(book, position, market)
                result.trades.append(trade)
                realised += trade.pnl

            for position in book.positions:
                if self.assignment.assigned(position, chain, spot):
                    trade = self._assign(book, position, chain, spot, day)
                    result.trades.append(trade)
                    realised += trade.pnl

            for position in book.positions:
                for leg in position.legs:
                    marked += 1
                    quote = chain.quote(leg.contract)
                    if quote is None or not quote.has_quote:
                        unquoted += 1

            if self.intraday_stop and market.enricher is not None and book.positions:
                # Today's close IV is the latest vol observable *before*
                # tomorrow's low, so it is what tomorrow's stress pass may use.
                prior_iv = leg_ivs(chain, market.greeks(day), book.positions)

            result.equity.append(
                EquityPoint(
                    date=day,
                    spot=spot,
                    cash=book.cash,
                    market_value=book.market_value(chain, spot, self.fill),
                    realised=realised,
                    unrealised=book.unrealised(chain, spot, self.fill),
                    open_positions=len(book),
                )
            )

        result.open_at_end = book.positions
        result.quality = DataQuality(
            sessions=len(sessions),
            sessions_without_chain=no_chain,
            marked_leg_days=marked,
            unquoted_leg_days=unquoted,
            divergent_sessions=len(divergent),
            decisions_on_divergent_sessions=sum(
                (t.opened_on in divergent) + (t.closed_on in divergent) for t in result.trades
            ),
        )
        return result

    # -- one order at a time -----------------------------------------------

    def _open(
        self, book: Book, order: Open, chain: ChainSnapshot, spot: float, day: dt.date
    ) -> Declined | None:
        unit: list[Leg] = []
        unit_cash = 0.0
        unit_spread = 0.0
        for leg_order in order.legs:
            quote = chain.quote(leg_order.contract)
            if quote is None or not quote.has_quote:
                # Entry is the one place the intrinsic fallback is wrong: you
                # cannot open a position nobody is quoting, at any price.
                return Declined(day, order.tag, f"no two-sided quote for {leg_order.contract}")
            side = BUY if leg_order.ratio > 0 else SELL
            price = self.fill.price(quote, side, intrinsic=leg_order.contract.intrinsic(spot))
            unit.append(Leg(leg_order.contract, leg_order.ratio, price))
            unit_cash += -leg_order.ratio * price * self.multiplier
            unit_spread += self.fill.haircut_on(quote) * abs(leg_order.ratio) * self.multiplier

        unit_margin = self.margin.requirement(unit, spot, self.multiplier)
        units = self.sizing.units(
            unit_cash=unit_cash,
            unit_margin=unit_margin,
            available_cash=book.cash - self._margin_used(book, spot),
            max_units=order.max_units,
        )
        if units < 1:
            return Declined(day, order.tag, "sizing declined: capital or unit limit")

        legs = tuple(Leg(leg.contract, leg.quantity * units, leg.entry_price) for leg in unit)
        contracts = sum(abs(leg.quantity) for leg in legs)
        commission = self.fill.commission_on(contracts)
        book.add(
            Position(
                id=book.next_id(),
                legs=legs,
                opened_on=day,
                entry_spot=spot,
                entry_cash=unit_cash * units - commission,
                units=units,
                entry_margin=unit_margin * units,
                entry_costs=Costs(unit_spread * units, commission),
                tag=order.tag,
            )
        )
        return None

    def _close(
        self, book: Book, order: Close, chain: ChainSnapshot, spot: float, day: dt.date
    ) -> Trade | Declined:
        try:
            position = book.get(order.position_id)
        except LookupError:
            # A strategy asking twice, or asking about something already
            # settled, is a strategy bug worth reporting -- not a crash that
            # loses the rest of the run.
            return Declined(day, order.reason, f"position {order.position_id} is not open")

        exit_cash, exit_costs = position.close_out(chain, spot, self.fill, self.multiplier)
        book.remove(position.id, exit_cash)
        return _trade(
            position,
            closed_on=day,
            exit_cash=exit_cash,
            reason=order.reason,
            exit_costs=exit_costs,
        )

    def _settle(self, book: Book, position: Position, market: MarketView) -> Trade:
        spot = settlement_spot(market, position.expiration)
        exit_cash, exit_costs = position.settle_out(spot, self.fill, self.multiplier)
        book.remove(position.id, exit_cash)
        return _trade(
            position,
            closed_on=position.expiration,
            exit_cash=exit_cash,
            reason="expiration",
            exit_costs=exit_costs,
        )

    def _assign(
        self, book: Book, position: Position, chain: ChainSnapshot, spot: float, day: dt.date
    ) -> Trade:
        exit_cash, exit_costs = position.assign_out(chain, spot, self.fill, self.multiplier)
        book.remove(position.id, exit_cash)
        return _trade(
            position,
            closed_on=day,
            exit_cash=exit_cash,
            reason="early_assignment",
            exit_costs=exit_costs,
        )

    def _intraday(
        self,
        book: Book,
        market: MarketView,
        strategy: Strategy,
        day: dt.date,
        chain: ChainSnapshot,
        spot: float,
        prior_iv: dict[Contract, float],
    ) -> tuple[list[Trade], dict[Contract, float]]:
        """Ask again at the session's low, and act only on a stop.

        Only positions opened on an **earlier** session are eligible: one
        opened at today's close could not have stopped out at a low that
        happened before it existed.
        """
        enricher = market.enricher
        if enricher is None:
            raise ValueError(
                "intraday_stop needs solved volatilities; load the MarketView "
                "with enrich=Enricher(...)"
            )
        bar = market.bar(day)
        stressable = eligible(book.positions, day)
        if not stressable or not (bar.low < spot):
            return [], prior_iv

        stressed = stress_chain(
            chain,
            stressable,
            low=bar.low,
            day=day,
            prior_iv=prior_iv,
            model=enricher.model,
            rate=enricher.rate,
            dividend_yield=enricher.dividend_yield,
        )
        context = Context(
            day=day,
            market=market.pin(day),
            book=book.view(),
            chain=stressed,
            spot=replace(bar, close=bar.low),
            stressed=True,
        )

        out: list[Trade] = []
        for order in strategy.on_day(context):
            # A take-profit or a time exit at the low would be a limit order
            # we cannot assume filled there. Only the stop is triggerable by
            # the market coming to you.
            if not isinstance(order, Close) or "stop" not in order.reason:
                continue
            outcome = self._close(book, order, stressed, bar.low, day)
            if isinstance(outcome, Trade):
                # Relabelled so the exit reason says which bound produced it:
                # a `stop_loss_intraday` never appears in the close-only run.
                out.append(replace(outcome, reason=f"{order.reason}_intraday"))
        return out, prior_iv

    def _margin_used(self, book: Book, spot: float) -> float:
        """Capital every open position ties up, recomputed at today's spot."""
        return sum(self.margin.requirement(p.legs, spot, self.multiplier) for p in book.positions)


def _tag_of(order: Close | Open) -> str:
    return order.tag if isinstance(order, Open) else order.reason


def _trade(
    position: Position,
    *,
    closed_on: dt.date,
    exit_cash: float,
    reason: str,
    exit_costs: Costs,
) -> Trade:
    return Trade(
        position_id=position.id,
        tag=position.tag,
        legs=position.legs,
        units=position.units,
        opened_on=position.opened_on,
        closed_on=closed_on,
        expiration=position.expiration,
        entry_cash=position.entry_cash,
        exit_cash=exit_cash,
        entry_margin=position.entry_margin,
        reason=reason,
        costs=position.entry_costs + exit_costs,
    )
