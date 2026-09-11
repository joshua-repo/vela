"""`vela` -- run one backtest from the command line.

    vela SYMBOL --start 2020-01-02 --end 2020-12-31
    vela SYMBOL --start ... --end ... --iv-rank 0.5 --breakeven

The layering rule holds here: this imports the rest of vela, and nothing in
vela imports it. A runner is a consumer of the engine.

**It refuses rather than guesses.** A symbol whose data is not cached does not
silently fetch an hour of vendor requests inside a backtest -- it stops and
prints the exact `antlia-history` commands to run first. Warming a cache is a
batch job, and one that costs money on a metered plan.

Two defaults are opinionated, for reasons the project has measured:

- **A gated run also runs its null.** A gate that does not beat always-entering
  has demonstrated nothing, and printing only the gated number invites the
  opposite conclusion.
- **Costs are not zero.** The fill model crosses half a half-spread and charges
  a commission unless told otherwise; `--haircut 0` is a diagnostic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from vela.data.guards import CorporateActionSuspected
from vela.data.market import MarketView
from vela.engine.loop import BacktestEngine
from vela.engine.results import Result
from vela.execution.fill import FillModel
from vela.metrics.curve import curve_stats
from vela.metrics.trades import TradeStats, trade_stats
from vela.pricing.enrich import Enricher
from vela.risk.margin import MaxLossMargin
from vela.risk.sizing import FixedRisk
from vela.run.artifacts import write_run
from vela.strategy.gate import AlwaysOpen, Gate, IVRank
from vela.strategy.vertical import VerticalSpread

#: Haircuts probed by `--breakeven`, in half-spreads.
_PROBES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vela",
        description="Backtest a vertical spread on one symbol, from cached antlia data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("symbol", help="underlying root, e.g. the ticker")
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end", required=True, metavar="YYYY-MM-DD")
    parser.add_argument(
        "--as-of",
        default=None,
        metavar="TIMESTAMP",
        help="read the store as it stood at this moment (ISO 8601), so a "
        "result survives a vendor restatement; default is the newest data",
    )
    parser.add_argument(
        "--max-dte",
        type=int,
        default=45,
        help="must match the --max-dte the cache was filled with, or antlia "
        "reports the whole window uncovered",
    )

    shape = parser.add_argument_group("structure")
    shape.add_argument("--right", choices=["P", "C"], default="P", help="P: credit, C: debit")
    shape.add_argument("--moneyness", type=float, default=0.95, help="short strike / spot")
    shape.add_argument("--width", type=float, default=5.0, help="dollars between strikes")
    shape.add_argument("--entry-dte", type=int, nargs=2, default=[30, 45], metavar=("LO", "HI"))

    exits = parser.add_argument_group("exits")
    exits.add_argument("--take-profit", type=float, default=0.5, help="fraction of max profit")
    exits.add_argument("--stop-loss", type=float, default=0.5, help="fraction of max loss")
    exits.add_argument("--exit-dte", type=int, default=7, help="close at this DTE; -1 to disable")

    book = parser.add_argument_group("book")
    book.add_argument("--max-concurrent", type=int, default=3)
    book.add_argument("--spacing", type=int, default=7, help="min days between entries")
    book.add_argument("--cash", type=float, default=100_000.0)
    book.add_argument("--risk", type=float, default=1_000.0, help="capital at risk per entry")

    signal = parser.add_argument_group("signal")
    signal.add_argument(
        "--iv-rank",
        type=float,
        default=None,
        metavar="MIN",
        help="gate entries on ATM IV rank; omit for the no-signal null",
    )
    signal.add_argument("--iv-lookback", type=int, default=120)
    signal.add_argument("--iv-warmup", type=int, default=20)

    costs = parser.add_argument_group("execution")
    costs.add_argument("--haircut", type=float, default=0.5, help="half-spreads crossed per leg")
    costs.add_argument("--commission", type=float, default=0.65, help="dollars per contract")
    costs.add_argument(
        "--blackout",
        type=float,
        default=None,
        metavar="FRAC",
        help="do not trade on sessions whose 17:15 mid is further than this "
        "from the 16:00 close (earnings dates); omit to trade through them",
    )
    costs.add_argument("--rate", type=float, default=0.04)
    costs.add_argument("--dividend-yield", type=float, default=0.0)

    out = parser.add_argument_group("output")
    out.add_argument("--out", default="runs", help="artifact root; '-' writes nothing")
    out.add_argument("--label", default=None)
    out.add_argument(
        "--envelope",
        action="store_true",
        help="also run the pessimistic bound: re-ask at each session's low and "
        "act on a stop. The width of the band is the finding",
    )
    out.add_argument(
        "--breakeven",
        action="store_true",
        help="also report the haircut at which this configuration breaks even",
    )
    out.add_argument(
        "--fetch",
        action="store_true",
        help="allow antlia to fetch missing data (metered, slow); off by default",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.fetch:
        # Provably offline: a covered read costs no vendor requests and never
        # authenticates. Set before antlia is touched.
        os.environ["ANTLIA_HISTORY_FETCH"] = "0"

    enricher = Enricher(rate=args.rate, dividend_yield=args.dividend_yield)
    try:
        market = MarketView.load(
            args.symbol,
            args.start,
            args.end,
            max_dte=args.max_dte,
            enrich=enricher,
            as_of=_as_of(args.as_of),
        )
    except CorporateActionSuspected as exc:
        print(f"refusing to run:\n{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 -- antlia raises NotCovered by name
        _explain_missing_data(args, exc)
        return 1

    engine = BacktestEngine(
        fill=FillModel(haircut=args.haircut, commission=args.commission),
        margin=MaxLossMargin(),
        sizing=FixedRisk(args.risk),
        initial_cash=args.cash,
        blackout=args.blackout,
    )

    gated = _run(market, engine, args, _gate(args))
    print(_header(args, market))
    if args.iv_rank is None:
        _report("no gate (null)", gated)
    else:
        _report(f"IVRank(>={args.iv_rank:g})", gated)
        _report("no gate (null)", _run(market, engine, args, AlwaysOpen()))
        print("\n  A gated result that does not beat its own null has shown nothing.")

    _report_quality(gated)

    if args.envelope:
        _report_envelope(market, engine, args)

    if args.breakeven:
        _report_breakeven(market, args)

    if args.out != "-":
        strategy = _strategy(args, _gate(args))
        path = write_run(
            gated,
            Path(args.out) / args.symbol.lower(),
            market=market,
            engine=engine,
            strategy=strategy,
            label=args.label,
        )
        print(f"\nartifacts: {path}")
    return 0


# -- assembly -----------------------------------------------------------------


def _as_of(raw: str | None) -> dt.datetime | None:
    """An aware timestamp from the flag. Naive input is read as UTC.

    antlia stamps every append in UTC, so a naive timestamp compared against
    them would silently be off by the local offset -- and a provenance pin
    that is quietly wrong is worse than none.
    """
    if raw is None:
        return None
    parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def _gate(args: argparse.Namespace) -> Gate:
    if args.iv_rank is None:
        return AlwaysOpen()
    return IVRank(minimum=args.iv_rank, lookback=args.iv_lookback, warmup=args.iv_warmup)


def _strategy(args: argparse.Namespace, gate: Gate) -> VerticalSpread:
    return VerticalSpread(
        right=args.right,
        moneyness=args.moneyness,
        width=args.width,
        entry_dte=(args.entry_dte[0], args.entry_dte[1]),
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        exit_dte=None if args.exit_dte < 0 else args.exit_dte,
        gate=gate,
        max_concurrent=args.max_concurrent,
        min_days_between_entries=args.spacing,
        tag="vertical",
    )


def _run(
    market: MarketView, engine: BacktestEngine, args: argparse.Namespace, gate: Gate
) -> Result:
    # A fresh strategy per run: a gate carries observation history that must
    # not leak between configurations.
    return engine.run(market, _strategy(args, gate))


# -- reporting ----------------------------------------------------------------


def _header(args: argparse.Namespace, market: MarketView) -> str:
    first, last = market.sessions[0], market.sessions[-1]
    move = market.bar(last).close / market.bar(first).close - 1
    return (
        f"\n{args.symbol}  {first} .. {last}  ({len(market.sessions)} sessions, "
        f"underlying {move:+.1%})\n"
        f"  data {market.version}\n"
        f"  {args.right} vertical, short at {args.moneyness:g}x spot, ${args.width:g} wide, "
        f"{args.entry_dte[0]}-{args.entry_dte[1]} DTE\n"
        f"  fills cross {args.haircut:g} half-spreads at ${args.commission:g}/contract\n"
    )


def _report(label: str, result: Result) -> None:
    stats = trade_stats(result.trades)
    if stats.n == 0:
        print(f"  {label:<20} no trades")
        return
    curve = curve_stats(result.equity, initial=result.initial_cash)
    lo, hi = stats.win_rate_ci
    print(
        f"  {label:<20} n={stats.n:<3} win {stats.win_rate:>5.1%} [{lo:.2f},{hi:.2f}]  "
        f"expect ${stats.expectancy:>8,.2f}  t={stats.t_stat:>6.2f}  "
        f"net ${stats.total_pnl:>9,.0f}  maxDD {curve.max_drawdown:>6.1%}"
    )
    print(
        f"  {'':<20} gross ${stats.gross_pnl:>9,.0f}  spread ${-stats.costs.spread:>9,.0f}  "
        f"commission ${-stats.costs.commission:>7,.0f}   {_verdict(stats)}"
    )


def _verdict(stats: TradeStats) -> str:
    if stats.n < 30:
        return f"n={stats.n} is too small to conclude anything"
    return "significant" if stats.significant else "indistinguishable from zero"


def _report_envelope(market: MarketView, engine: BacktestEngine, args: argparse.Namespace) -> None:
    """Two runs bracketing the answer EOD data cannot give.

    A stop is path-dependent and a daily bar has no path, so close-only
    marking is optimistic for it. Neither bound is the truth and they must
    never be ranked against each other -- **the width between them is the
    finding**. Narrow means EOD granularity is not what limits this result.
    """
    optimistic = trade_stats(_run(market, engine, args, _gate(args)).trades)
    pessimistic = trade_stats(
        _run(market, replace(engine, intraday_stop=True), args, _gate(args)).trades
    )
    band = pessimistic.total_pnl - optimistic.total_pnl
    print("\n  envelope (a bound each, never a ranking):")
    for label, stats in (
        ("close-only, optimistic", optimistic),
        ("intraday,   pessimistic", pessimistic),
    ):
        print(f"    {label:<26} n={stats.n:<3} ${stats.total_pnl:>10,.0f}")
    width = abs(band) / max(abs(optimistic.total_pnl), 1.0)
    tail = " (inverted: stopping earlier helped)" if band > 0 else ""
    print(f"    -> band ${abs(band):,.0f}, {width:.0%} of the close-only result{tail}")


def _report_quality(result: Result) -> None:
    """Print what the run noticed about its own data.

    Always, including when every number is zero. "Checked, and clean" and
    "never checked" are different claims and should not print the same.
    """
    quality = result.quality
    if quality is None:
        return
    print(
        f"\n  data: {quality.sessions} sessions, "
        f"{quality.sessions_without_chain} without a chain; "
        f"{quality.unquoted_leg_days} of {quality.marked_leg_days} leg-days marked at "
        f"intrinsic ({quality.unquoted_rate:.1%})"
    )
    if quality.divergent_sessions:
        blocked = sum(1 for d in result.declined if d.reason.startswith("blackout"))
        note = f", {blocked} orders blacked out" if blocked else ""
        print(
            f"        {quality.divergent_sessions} sessions where the 17:15 quote "
            f"disagrees with the close (earnings); "
            f"{quality.decisions_on_divergent_sessions} trade decisions landed on "
            f"one{note}"
        )


def _report_breakeven(market: MarketView, args: argparse.Namespace) -> None:
    """The haircut at which this configuration stops making money.

    More informative than the P&L at one assumed haircut, because the
    assumption is what dominates: measured on this strategy, crossing the
    spread cost roughly four times the commission and three times the gross
    edge. The break-even says how good an execution you would need.
    """
    print("\n  break-even haircut (half-spreads crossed per leg):")
    results = []
    for haircut in _PROBES:
        engine = BacktestEngine(
            fill=FillModel(haircut=haircut, commission=args.commission),
            margin=MaxLossMargin(),
            sizing=FixedRisk(args.risk),
            initial_cash=args.cash,
            blackout=args.blackout,
        )
        stats = trade_stats(_run(market, engine, args, _gate(args)).trades)
        results.append((haircut, stats.total_pnl))
        print(f"    {haircut:>5.2f}   ${stats.total_pnl:>10,.0f}")

    crossing = _interpolate_zero(results)
    if crossing is None:
        verdict = (
            "never positive -- the edge does not survive any crossing cost"
            if results[0][1] <= 0
            else "positive at every haircut probed"
        )
        print(f"    -> {verdict}")
    else:
        print(
            f"    -> breaks even at {crossing:.2f} half-spreads, i.e. fills within "
            f"{crossing / 2:.0%} of the full spread from the mid"
        )


def _interpolate_zero(points: list[tuple[float, float]]) -> float | None:
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if y0 > 0 >= y1:
            return x0 + (x1 - x0) * y0 / (y0 - y1)
    return None


def _explain_missing_data(args: argparse.Namespace, exc: Exception) -> None:
    """Say exactly what to run. Never fetch an hour of data from inside a run."""
    print(f"cannot run: {exc}\n", file=sys.stderr)
    print("Warm the cache first. `plan` costs no vendor requests -- run it:", file=sys.stderr)
    print(
        f"\n  uv run antlia-history plan option_eod {args.symbol} "
        f"--start {args.start} --end {args.end} --max-dte {args.max_dte}\n"
        f"  uv run antlia-history fill equity_eod {args.symbol} "
        f"--start {args.start} --end {args.end}\n"
        f"  uv run antlia-history fill option_eod {args.symbol} "
        f"--start {args.start} --end {args.end} --max-dte {args.max_dte} --workers 2\n",
        file=sys.stderr,
    )
    print(
        "Budget roughly 75 seconds per request -- a year of one symbol's chain is\n"
        "about an hour. Run ONE antlia-history command at a time: a second one\n"
        "invalidates the first's ThetaData session and kills it.\n"
        "\n"
        "If the fill stalls at the same request count twice, a scope is permanently\n"
        "broken (ThetaData returns malformed rows for not-yet-expired contracts).\n"
        "End the window 45 days before the earliest unfilled expiration.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
