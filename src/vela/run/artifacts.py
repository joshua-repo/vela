"""What a run leaves behind.

Three files per run, and the division between them is the point:

- `trades.csv` -- one row per position. The decisions.
- `equity.csv` -- one row per session. The account.
- `summary.json` -- everything needed to say what was run, and what came out.

**`summary.json` is the layer downstream reads**, pictor included. Per-run
CSVs are the source of truth for a single run; the summary is what aggregates
across runs without re-parsing any of them. A dashboard built on the CSVs has
to know each strategy's column set; one built on the summary does not.

Runs are **append-only**. A run directory is stamped with the moment it ran
and never overwritten, because a result whose inputs you can no longer
reconstruct is not evidence of anything. Two runs that finish in the same
second with the same label -- which is the normal case for parallel cells --
get `-1`, `-2` suffixes; `mkdir` is atomic, so no lock is needed.
"""

from __future__ import annotations

import csv
import datetime as dt
import itertools
import json
import math
import os
import subprocess
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from vela.data.market import MarketView
from vela.engine.loop import BacktestEngine
from vela.engine.results import Result
from vela.metrics.curve import curve_stats
from vela.metrics.trades import trade_stats
from vela.strategy.base import Strategy

TRADE_COLUMNS = (
    "position_id",
    "tag",
    "opened_on",
    "closed_on",
    "expiration",
    "held_days",
    "units",
    "reason",
    "entry_cash",
    "exit_cash",
    "pnl",
    "entry_margin",
    "return_on_margin",
    "gross_pnl",
    "cost_spread",
    "cost_commission",
    "legs",
)

EQUITY_COLUMNS = (
    "date",
    "spot",
    "cash",
    "market_value",
    "equity",
    "realised",
    "unrealised",
    "open_positions",
    "interest",
)


def write_run(
    result: Result,
    directory: Path | str,
    *,
    market: MarketView,
    engine: BacktestEngine,
    strategy: Strategy,
    label: str | None = None,
    invocation: dict[str, Any] | None = None,
) -> Path:
    """Write one run under `directory`, returning the directory it created."""
    stamp = _stamp()
    name = f"{stamp}_{label}" if label else stamp
    out = _claim(Path(directory), name)

    _write_trades(result, out / "trades.csv")
    _write_equity(result, out / "equity.csv")
    (out / "summary.json").write_text(
        json.dumps(
            summary(result, market=market, engine=engine, strategy=strategy, invocation=invocation),
            indent=2,
        )
        + "\n"
    )
    return out


def summary(
    result: Result,
    *,
    market: MarketView,
    engine: BacktestEngine,
    strategy: Strategy,
    invocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The provenance record: what ran, over what, and what came out.

    `invocation` is how the run was asked for -- the command line and every
    parsed argument, defaults included. The rest of the record describes the
    objects that ran, which is not the same thing: `--rate`, `--max-dte` and
    `--as-of` change the result without surfacing on any of them. A reader
    that needs to know whether two runs differ in exactly one input compares
    these. Absent, not empty, when a caller built the run without a CLI.
    """
    trades = trade_stats(result.trades)
    curve = (
        curve_stats(result.equity, initial=result.initial_cash, risk_free=engine.cash_rate)
        if len(result.equity) > 1
        else None
    )
    declines: dict[str, int] = {}
    for decline in result.declined:
        key = decline.reason.split(":")[0]
        declines[key] = declines.get(key, 0) + 1

    record: dict[str, Any] = {
        "run_at": dt.datetime.now(dt.UTC).isoformat(),
        "git": _git(),
        "data": {
            "source": "antlia",
            "symbol": market.symbol,
            "start": result.equity[0].date if result.equity else None,
            "end": result.equity[-1].date if result.equity else None,
            "sessions": len(result.equity),
            # The regime in one number: first to last close over the window.
            # A result has to be read beside it -- selling puts through a
            # rising market says little about selling puts.
            "underlying_move": (
                result.equity[-1].spot / result.equity[0].spot - 1
                if len(result.equity) > 1 and result.equity[0].spot
                else None
            ),
        },
        "engine": {
            "initial_cash": engine.initial_cash,
            "cash_rate": engine.cash_rate,
            "multiplier": engine.multiplier,
            "fill": _params(engine.fill),
            "assignment": _params(engine.assignment),
            "margin": _params(engine.margin),
            "sizing": _params(engine.sizing),
        },
        "strategy": {"name": strategy.name, "params": _params(strategy)},
        "results": {
            "trades": asdict(trades),
            "curve": asdict(curve) if curve else None,
            # Account-level: no trade carries it, and `curve` measures the
            # excess over it.
            "interest": result.interest,
            "open_at_end": len(result.open_at_end),
            "declined": declines,
        },
    }
    if invocation is not None:
        record["invocation"] = invocation
    cleaned: dict[str, Any] = _clean(record)
    return cleaned


def _write_trades(result: Result, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRADE_COLUMNS)
        writer.writeheader()
        for trade in result.trades:
            writer.writerow(
                {
                    "position_id": trade.position_id,
                    "tag": trade.tag,
                    "opened_on": trade.opened_on,
                    "closed_on": trade.closed_on,
                    "expiration": trade.expiration,
                    "held_days": trade.held_days,
                    "units": trade.units,
                    "reason": trade.reason,
                    "entry_cash": round(trade.entry_cash, 4),
                    "exit_cash": round(trade.exit_cash, 4),
                    "pnl": round(trade.pnl, 4),
                    "entry_margin": round(trade.entry_margin, 4),
                    "return_on_margin": _round(trade.return_on_margin, 6),
                    "gross_pnl": round(trade.gross_pnl, 4),
                    "cost_spread": round(trade.costs.spread, 4),
                    "cost_commission": round(trade.costs.commission, 4),
                    "legs": " ".join(
                        f"{leg.quantity:+d}@{leg.contract.strike:g}{leg.contract.right}"
                        f":{leg.entry_price:g}"
                        for leg in trade.legs
                    ),
                }
            )


def _write_equity(result: Result, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EQUITY_COLUMNS)
        writer.writeheader()
        for point in result.equity:
            writer.writerow(
                {
                    "date": point.date,
                    "spot": round(point.spot, 4),
                    "cash": round(point.cash, 4),
                    "market_value": round(point.market_value, 4),
                    "equity": round(point.equity, 4),
                    "realised": round(point.realised, 4),
                    "unrealised": round(point.unrealised, 4),
                    "open_positions": point.open_positions,
                    "interest": round(point.interest, 4),
                }
            )


def _params(obj: object) -> dict[str, Any]:
    """A component's configuration: its dataclass fields, or just its name.

    Private and derived fields are dropped -- a strategy's internal signal
    state is not a parameter of the run, and recording it would make two
    identical configurations look different.
    """
    record: dict[str, Any] = {"type": type(obj).__name__}
    if is_dataclass(obj) and not isinstance(obj, type):
        for field in fields(obj):
            if field.name.startswith("_") or field.name == "name":
                continue
            value = getattr(obj, field.name)
            # A component can hold another -- a strategy holds its gate, which
            # holds its own parameters. Recording only the class name would
            # lose the numbers that produced the result.
            record[field.name] = _params(value) if _is_component(value) else value
    return record


def _is_component(value: object) -> bool:
    """Whether a parameter is itself a configured object rather than data."""
    return not isinstance(value, (str, bytes, int, float, bool, type(None), list, tuple, dict))


#: Overrides the default runs root. `--out` overrides both.
RUNS_ENV = "VELA_RUNS"


def runs_root() -> Path:
    """Where runs land when nobody says otherwise: `$VELA_RUNS`, else
    `~/.vela/runs`.

    Fixed rather than relative to the working directory, because pictor reads
    this folder and cannot follow a result to wherever its runner happened to
    stand -- the same reason antlia keeps its store at `~/.antlia/store`.
    """
    raw = os.environ.get(RUNS_ENV)
    return Path(raw).expanduser() if raw else Path.home() / ".vela" / "runs"


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _claim(directory: Path, name: str) -> Path:
    """Create a fresh run directory, suffixing `name` until one is free."""
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in itertools.count():
        out = directory / (name if attempt == 0 else f"{name}-{attempt}")
        try:
            out.mkdir()
        except FileExistsError:
            continue
        return out
    raise AssertionError("unreachable")


#: Where vela's own source lives. git is asked about *this* checkout, not the
#: caller's working directory: a driver launching cells from another repo
#: would otherwise stamp every result with the driver's commit.
_SOURCE = Path(__file__).resolve().parent


def _git() -> dict[str, Any]:
    """The commit a run was produced at, and whether the tree was dirty.

    A dirty tree is recorded rather than refused: research happens on dirty
    trees. What matters is that the result says so. An install that is not a
    checkout records nulls rather than whatever repository surrounds it.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=_SOURCE
        ).stdout.strip()

    try:
        # Installed into another project's venv, the nearest repository is that
        # project's. Only a checkout whose src/vela is this package counts.
        top = Path(git("rev-parse", "--show-toplevel")).resolve()
        if top / "src" / "vela" != _SOURCE.parent:
            return {"sha": None, "dirty": None}
        sha = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "dirty": None}


def _round(value: float, digits: int) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else round(value, digits)


def _clean(value: Any) -> Any:
    """JSON-safe: dates as ISO strings, NaN and infinity as null.

    `json.dumps` writes bare `NaN` and `Infinity`, which are not JSON and
    which every strict parser downstream rejects. A metric that has no value
    is null.
    """
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
