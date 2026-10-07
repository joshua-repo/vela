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

Every default comes from `vela.run.config.RunConfig`, so a run assembled any
other way -- a combo cell, a test -- is the same run this command would make.

**Exit codes say what to do next**, so a driver running many of these can act
without parsing stderr:

    0  ran
    1  crashed -- a bug, with its traceback; never reported as missing data
    2  usage error (argparse)
    3  data missing: the store does not hold or cannot vouch for the window.
       Warm the cache, then re-run
    4  refused: a corporate action inside the window. Change the window
    5  denied: the vendor will never serve this window. Change the window
    6  fetch failed (only with --fetch). Transient; retry
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from vela.data.guards import CorporateActionSuspected
from vela.data.market import MarketView, data_failure
from vela.engine.loop import BacktestEngine
from vela.engine.results import Result
from vela.metrics.curve import curve_stats
from vela.metrics.trades import TradeStats, trade_stats
from vela.run.artifacts import runs_root, write_run
from vela.run.config import DEFAULTS as D
from vela.run.config import RunConfig
from vela.strategy.gate import AlwaysOpen, Gate

#: Haircuts probed by `--breakeven`, in half-spreads.
_PROBES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)

EXIT_OK = 0
EXIT_MISSING = 3
EXIT_REFUSED = 4
EXIT_DENIED = 5
EXIT_FETCH_FAILED = 6
_EXIT_FOR = {"missing": EXIT_MISSING, "denied": EXIT_DENIED, "fetch": EXIT_FETCH_FAILED}


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
        default=D["max_dte"],
        help="must match the --max-dte the cache was filled with, or antlia "
        "reports the whole window uncovered",
    )

    shape = parser.add_argument_group("structure")
    shape.add_argument(
        "--right", choices=["P", "C"], default=D["right"], help="P: credit, C: debit"
    )
    shape.add_argument(
        "--moneyness", type=float, default=D["moneyness"], help="short strike / spot"
    )
    shape.add_argument("--width", type=float, default=D["width"], help="dollars between strikes")
    shape.add_argument(
        "--entry-dte", type=int, nargs=2, default=list(D["entry_dte"]), metavar=("LO", "HI")
    )

    exits = parser.add_argument_group("exits")
    exits.add_argument(
        "--take-profit", type=float, default=D["take_profit"], help="fraction of max profit"
    )
    exits.add_argument(
        "--stop-loss", type=float, default=D["stop_loss"], help="fraction of max loss"
    )
    exits.add_argument(
        "--exit-dte",
        type=int,
        default=-1 if D["exit_dte"] is None else D["exit_dte"],
        help="close at this DTE; -1 to disable",
    )

    book = parser.add_argument_group("book")
    book.add_argument("--max-concurrent", type=int, default=D["max_concurrent"])
    book.add_argument("--spacing", type=int, default=D["spacing"], help="min days between entries")
    book.add_argument("--cash", type=float, default=D["cash"])
    book.add_argument("--risk", type=float, default=D["risk"], help="capital at risk per entry")
    book.add_argument(
        "--cash-rate",
        type=float,
        default=D["cash_rate"],
        help="annual rate cash earns (ACT/365), and the benchmark excess return "
        "and Sharpe are measured against; default: --rate",
    )

    signal = parser.add_argument_group("signal")
    signal.add_argument(
        "--iv-rank",
        type=float,
        default=D["iv_rank"],
        metavar="MIN",
        help="gate entries on ATM IV rank; omit for the no-signal null",
    )
    signal.add_argument("--iv-lookback", type=int, default=D["iv_lookback"])
    signal.add_argument("--iv-warmup", type=int, default=D["iv_warmup"])

    costs = parser.add_argument_group("execution")
    costs.add_argument(
        "--haircut", type=float, default=D["haircut"], help="half-spreads crossed per leg"
    )
    costs.add_argument(
        "--commission", type=float, default=D["commission"], help="dollars per contract"
    )
    costs.add_argument(
        "--blackout",
        type=float,
        default=D["blackout"],
        metavar="FRAC",
        help="do not trade on sessions whose 17:15 mid is further than this "
        "from the 16:00 close (earnings dates); omit to trade through them",
    )
    costs.add_argument("--rate", type=float, default=D["rate"])
    costs.add_argument("--dividend-yield", type=float, default=D["dividend_yield"])

    out = parser.add_argument_group("output")
    out.add_argument(
        "--out",
        default=None,
        help="artifact root; '-' writes nothing (default: $VELA_RUNS, else ~/.vela/runs)",
    )
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


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse, then resolve the defaults that depend on other flags or the
    environment -- here, so that `invocation` records what actually ran (the
    rate, the directory), not None."""
    args = build_parser().parse_args(argv)
    if args.cash_rate is None:
        args.cash_rate = args.rate
    if args.out is None:
        args.out = str(runs_root())
    return args


def config_from(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        as_of=_as_of(args.as_of),
        max_dte=args.max_dte,
        right=args.right,
        moneyness=args.moneyness,
        width=args.width,
        entry_dte=(args.entry_dte[0], args.entry_dte[1]),
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        exit_dte=None if args.exit_dte < 0 else args.exit_dte,
        max_concurrent=args.max_concurrent,
        spacing=args.spacing,
        cash=args.cash,
        risk=args.risk,
        iv_rank=args.iv_rank,
        iv_lookback=args.iv_lookback,
        iv_warmup=args.iv_warmup,
        haircut=args.haircut,
        commission=args.commission,
        blackout=args.blackout,
        rate=args.rate,
        dividend_yield=args.dividend_yield,
        cash_rate=args.cash_rate,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse(argv)

    if not args.fetch:
        # Provably offline: a covered read costs no vendor requests and never
        # authenticates. Set before antlia is touched.
        os.environ["ANTLIA_HISTORY_FETCH"] = "0"

    config = config_from(args)
    try:
        market = config.load()
    except CorporateActionSuspected as exc:
        print(f"refusing to run:\n{exc}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as exc:
        kind = data_failure(exc)
        if kind is None:
            raise  # a bug, not a data problem: exit 1 with its traceback
        if kind == "missing":
            _explain_missing_data(args, exc)
        else:
            print(f"cannot run: {exc}", file=sys.stderr)
        return _EXIT_FOR[kind]

    engine = config.engine()

    gated = _run(market, engine, config)
    print(_header(args, market))
    if args.iv_rank is None:
        _report("no gate (null)", gated, args.cash_rate)
    else:
        _report(f"IVRank(>={args.iv_rank:g})", gated, args.cash_rate)
        _report("no gate (null)", _run(market, engine, config, AlwaysOpen()), args.cash_rate)
        print("\n  A gated result that does not beat its own null has shown nothing.")

    _report_quality(gated)

    if args.envelope:
        _report_envelope(market, engine, config)

    if args.breakeven:
        _report_breakeven(market, config)

    if args.out != "-":
        strategy = config.strategy()
        path = write_run(
            gated,
            Path(args.out).expanduser() / args.symbol.lower(),
            market=market,
            engine=engine,
            strategy=strategy,
            label=args.label,
            invocation={
                "argv": list(sys.argv[1:] if argv is None else argv),
                "args": vars(args),
            },
        )
        print(f"\nartifacts: {path}")
    return EXIT_OK


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


def _run(
    market: MarketView, engine: BacktestEngine, config: RunConfig, gate: Gate | None = None
) -> Result:
    # A fresh strategy per run: a gate carries observation history that must
    # not leak between configurations.
    return engine.run(market, config.strategy(gate))


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
        f"  fills cross {args.haircut:g} half-spreads at ${args.commission:g}/contract; "
        f"cash earns {args.cash_rate:.2%}\n"
    )


def _report(label: str, result: Result, cash_rate: float) -> None:
    stats = trade_stats(result.trades)
    if stats.n == 0:
        print(f"  {label:<20} no trades")
        return
    curve = curve_stats(result.equity, initial=result.initial_cash, risk_free=cash_rate)
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
    sharpe = "n/a" if math.isnan(curve.sharpe) else f"{curve.sharpe:.2f}"
    print(
        f"  {'':<20} account {curve.total_return:>+6.2%}  cash alone "
        f"{curve.risk_free_return:>+6.2%}  excess {curve.excess_return:>+6.2%}  "
        f"interest ${result.interest:>7,.0f}  sharpe {sharpe}"
    )


def _verdict(stats: TradeStats) -> str:
    if stats.n < 30:
        return f"n={stats.n} is too small to conclude anything"
    return "significant" if stats.significant else "indistinguishable from zero"


def _report_envelope(market: MarketView, engine: BacktestEngine, config: RunConfig) -> None:
    """Two runs bracketing the answer EOD data cannot give.

    A stop is path-dependent and a daily bar has no path, so close-only
    marking is optimistic for it. Neither bound is the truth and they must
    never be ranked against each other -- **the width between them is the
    finding**. Narrow means EOD granularity is not what limits this result.
    """
    optimistic = trade_stats(_run(market, engine, config).trades)
    pessimistic = trade_stats(_run(market, replace(engine, intraday_stop=True), config).trades)
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


def _report_breakeven(market: MarketView, config: RunConfig) -> None:
    """The haircut at which this configuration stops making money.

    More informative than the P&L at one assumed haircut, because the
    assumption is what dominates: measured on this strategy, crossing the
    spread cost roughly four times the commission and three times the gross
    edge. The break-even says how good an execution you would need.
    """
    print("\n  break-even haircut (half-spreads crossed per leg):")
    results = []
    for haircut in _PROBES:
        stats = trade_stats(_run(market, config.engine(haircut=haircut), config).trades)
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
