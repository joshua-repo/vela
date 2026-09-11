"""Run artifacts: three files, and a summary anything can parse."""

from __future__ import annotations

import csv
import datetime as dt
import json

import pytest

from tests.conftest import Scripted, chain, market
from vela.engine.loop import BacktestEngine
from vela.execution.fill import FillModel
from vela.instrument import Contract
from vela.orders import LegOrder, Open
from vela.risk.margin import MaxLossMargin
from vela.run.artifacts import summary, write_run

EXPIRY = dt.date(2026, 2, 20)
SHORT = Contract("TEST", EXPIRY, 95.0, "P")
LONG = Contract("TEST", EXPIRY, 90.0, "P")
PATH = {dt.date(2026, 1, 5): 100.0, dt.date(2026, 1, 6): 100.0, EXPIRY: 100.0}


def quotes(spot: float):
    return [
        (95.0, "P", max(95 - spot, 0) + 2.00, max(95 - spot, 0) + 2.40),
        (90.0, "P", max(90 - spot, 0) + 1.00, max(90 - spot, 0) + 1.40),
    ]


def enter_once(ctx):
    if ctx.book.positions or ctx.day != min(PATH):
        return []
    return [Open(legs=(LegOrder(SHORT, -1), LegOrder(LONG, +1)), tag="bps")]


@pytest.fixture
def run():
    mv = market(PATH, {d: chain(d, {EXPIRY: quotes(s)}) for d, s in PATH.items()})
    strategy = Scripted(enter_once)
    strategy.name = "Scripted"
    engine = BacktestEngine(fill=FillModel(0.5, 0.65), margin=MaxLossMargin())
    return engine.run(mv, strategy), mv, engine, strategy


def test_write_run_creates_the_three_files(run, tmp_path) -> None:
    result, mv, engine, strategy = run
    out = write_run(result, tmp_path, market=mv, engine=engine, strategy=strategy, label="x")
    assert {p.name for p in out.iterdir()} == {"trades.csv", "equity.csv", "summary.json"}
    assert out.name.endswith("_x")


def test_a_run_never_overwrites_another(run, tmp_path) -> None:
    """Append-only: a result whose inputs you cannot reconstruct is not evidence."""
    result, mv, engine, strategy = run
    out = write_run(result, tmp_path, market=mv, engine=engine, strategy=strategy)
    with pytest.raises(FileExistsError):
        out.mkdir()


def test_rows_match_the_result(run, tmp_path) -> None:
    result, mv, engine, strategy = run
    out = write_run(result, tmp_path, market=mv, engine=engine, strategy=strategy)
    trades = list(csv.DictReader((out / "trades.csv").open()))
    equity = list(csv.DictReader((out / "equity.csv").open()))
    assert len(trades) == len(result.trades)
    assert len(equity) == len(result.equity)
    assert float(trades[0]["pnl"]) == pytest.approx(result.trades[0].pnl, abs=1e-4)


def strict(text: str):
    """Parse rejecting NaN and Infinity, which are not JSON."""

    def reject(constant: str):
        raise ValueError(f"non-JSON constant {constant!r} in summary")

    return json.loads(text, parse_constant=reject)


def test_summary_is_strict_json(run, tmp_path) -> None:
    result, mv, engine, strategy = run
    out = write_run(result, tmp_path, market=mv, engine=engine, strategy=strategy)
    strict((out / "summary.json").read_text())


def test_absent_metrics_are_null_not_nan(run) -> None:
    """An empty run has no win rate; `null` says so and 0.0 would lie."""
    _, mv, engine, strategy = run
    empty = BacktestEngine().run(mv, Scripted(lambda ctx: []))
    record = strict(json.dumps(summary(empty, market=mv, engine=engine, strategy=strategy)))
    assert record["results"]["trades"]["n"] == 0
    assert record["results"]["trades"]["win_rate"] is None


def test_summary_records_what_was_run(run) -> None:
    result, mv, engine, strategy = run
    record = summary(result, market=mv, engine=engine, strategy=strategy)
    assert record["data"]["symbol"] == "TEST"
    assert record["engine"]["fill"] == {"type": "FillModel", "haircut": 0.5, "commission": 0.65}
    assert record["engine"]["margin"]["type"] == "MaxLossMargin"
    assert record["results"]["trades"]["n"] == len(result.trades)
    assert "sha" in record["git"]


def test_every_engine_component_reaches_the_summary(run) -> None:
    """A component that changes results and is not recorded makes the run
    irreproducible from its own provenance."""
    result, mv, engine, strategy = run
    record = summary(result, market=mv, engine=engine, strategy=strategy)
    assert set(record["engine"]) >= {"fill", "assignment", "margin", "sizing"}
    assert record["engine"]["assignment"]["type"] == "ParityAssignment"


def test_a_nested_component_records_its_parameters(run) -> None:
    """A strategy holds a gate; the gate's numbers produced the result."""
    from vela.strategy.gate import IVRank
    from vela.strategy.vertical import VerticalSpread

    result, mv, engine, _ = run
    gated = VerticalSpread(
        right="P",
        moneyness=0.95,
        width=5.0,
        entry_dte=(30, 45),
        take_profit=0.5,
        stop_loss=0.5,
        gate=IVRank(0.5),
    )
    record = summary(result, market=mv, engine=engine, strategy=gated)
    assert record["strategy"]["params"]["gate"]["minimum"] == 0.5
    assert record["strategy"]["params"]["gate"]["type"] == "IVRank"


# -- the command line ---------------------------------------------------------


def test_cli_refuses_an_uncached_symbol_and_says_what_to_run(capsys) -> None:
    """Never fetch an hour of metered vendor data from inside a backtest."""
    from vela.run.cli import main

    code = main(["NOSUCHTICKER", "--start", "2026-01-01", "--end", "2026-02-01"])
    assert code == 1
    err = capsys.readouterr().err
    assert "antlia-history fill" in err
    assert "NOSUCHTICKER" in err
    assert "one antlia-history command at a time" in err.lower()


def test_cli_defaults_to_offline() -> None:
    """A run must not authenticate or spend requests unless asked."""
    import os

    from vela.run.cli import main

    os.environ.pop("ANTLIA_HISTORY_FETCH", None)
    main(["NOSUCHTICKER", "--start", "2026-01-01", "--end", "2026-02-01"])
    assert os.environ["ANTLIA_HISTORY_FETCH"] == "0"


def test_cli_parses_the_documented_flags() -> None:
    from vela.run.cli import build_parser

    args = build_parser().parse_args(
        ["XYZ", "--start", "2020-01-02", "--end", "2020-12-31", "--iv-rank", "0.5"]
    )
    assert args.symbol == "XYZ"
    assert args.iv_rank == 0.5
    assert args.haircut == 0.5  # costs are not free by default
    assert args.commission == 0.65
