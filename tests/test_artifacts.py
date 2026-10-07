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


def test_the_regime_is_recorded_beside_the_result(run) -> None:
    """First to last close. The fixture's path is flat, so the move is zero --
    a recorded zero, not an absent field."""
    result, mv, engine, strategy = run
    record = summary(result, market=mv, engine=engine, strategy=strategy)
    assert record["data"]["underlying_move"] == pytest.approx(0.0)


def test_a_run_records_how_it_was_asked_for(run, tmp_path) -> None:
    """Every parsed argument, defaults included: `--rate` or `--as-of` change a
    result without appearing on any object the rest of the summary describes."""
    from vela.run.cli import build_parser

    result, mv, engine, strategy = run
    argv = ["XYZ", "--start", "2020-01-02", "--end", "2020-12-31", "--iv-rank", "0.5"]
    args = vars(build_parser().parse_args(argv))
    out = write_run(
        result,
        tmp_path,
        market=mv,
        engine=engine,
        strategy=strategy,
        invocation={"argv": argv, "args": args},
    )
    record = strict((out / "summary.json").read_text())
    assert record["invocation"]["argv"] == argv
    assert record["invocation"]["args"]["iv_rank"] == 0.5
    assert record["invocation"]["args"]["rate"] == 0.04
    assert record["invocation"]["args"]["as_of"] is None


def test_a_run_built_without_a_cli_has_no_invocation(run) -> None:
    """Absent, not empty: nothing was invoked."""
    result, mv, engine, strategy = run
    assert "invocation" not in summary(result, market=mv, engine=engine, strategy=strategy)


def test_parallel_runs_with_one_label_never_collide(run, tmp_path, monkeypatch) -> None:
    """Cells of one combo finish in the same second under the same label. The
    second must not crash after its compute is done -- it gets a suffix."""
    from vela.run import artifacts

    monkeypatch.setattr(artifacts, "_stamp", lambda: "20260101T000000Z")
    result, mv, engine, strategy = run
    outs = [
        write_run(result, tmp_path, market=mv, engine=engine, strategy=strategy, label="cell")
        for _ in range(3)
    ]
    assert [o.name for o in outs] == [
        "20260101T000000Z_cell",
        "20260101T000000Z_cell-1",
        "20260101T000000Z_cell-2",
    ]


def test_provenance_is_vela_s_checkout_not_the_callers(run, tmp_path, monkeypatch) -> None:
    """A driver launching cells from elsewhere must not stamp them with its
    own commit, or with nothing."""
    from vela.run.artifacts import _git

    expected = _git()
    monkeypatch.chdir(tmp_path)
    assert _git() == expected


def test_interest_is_recorded_and_measured_against(tmp_path) -> None:
    mv = market(PATH, {d: chain(d, {EXPIRY: quotes(s)}) for d, s in PATH.items()})
    strategy = Scripted(enter_once)
    strategy.name = "Scripted"
    engine = BacktestEngine(fill=FillModel(0.5, 0.65), margin=MaxLossMargin(), cash_rate=0.05)
    result = engine.run(mv, strategy)
    out = write_run(result, tmp_path, market=mv, engine=engine, strategy=strategy)
    record = strict((out / "summary.json").read_text())
    assert record["engine"]["cash_rate"] == 0.05
    assert record["results"]["interest"] == pytest.approx(result.interest)
    assert record["results"]["curve"]["risk_free"] == 0.05
    assert record["results"]["curve"]["risk_free_return"] > 0
    with (out / "equity.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[-1]["interest"]) == pytest.approx(result.interest, abs=1e-4)


# -- the command line ---------------------------------------------------------


def test_cli_refuses_an_uncached_symbol_and_says_what_to_run(capsys) -> None:
    """Never fetch an hour of metered vendor data from inside a backtest."""
    from vela.run.cli import main

    code = main(["NOSUCHTICKER", "--start", "2026-01-01", "--end", "2026-02-01"])
    assert code == 3  # missing: warm the cache
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


def test_cash_rate_defaults_to_the_pricing_rate() -> None:
    """One risk-free number unless told otherwise, resolved at parse time so
    `invocation` records the rate that ran rather than None."""
    from vela.run.cli import parse

    base = ["XYZ", "--start", "2020-01-02", "--end", "2020-12-31"]
    assert parse(base).cash_rate == 0.04
    assert parse([*base, "--rate", "0.03"]).cash_rate == 0.03
    assert parse([*base, "--rate", "0.03", "--cash-rate", "0"]).cash_rate == 0.0


# -- one source of defaults ---------------------------------------------------


def test_the_cli_and_a_bare_config_are_the_same_run() -> None:
    """A combo cell built as `RunConfig(symbol, start, end)` must be exactly
    what `vela SYMBOL --start --end` runs. Assembled separately, the library's
    own structural defaults once ran 19 trades where the CLI ran 42."""
    from vela.run.cli import config_from, parse
    from vela.run.config import RunConfig

    base = ["XYZ", "--start", "2020-01-02", "--end", "2020-12-31"]
    assert config_from(parse(base)) == RunConfig("XYZ", "2020-01-02", "2020-12-31")


def test_every_config_field_is_a_flag() -> None:
    """A knob a combo can turn but `vela run` cannot is a cell no single run
    can reproduce."""
    from dataclasses import fields

    from vela.run.cli import parse
    from vela.run.config import RunConfig

    args = vars(parse(["XYZ", "--start", "2020-01-02", "--end", "2020-12-31"]))
    assert {f.name for f in fields(RunConfig)} <= set(args)


def test_a_config_builds_a_fresh_strategy_every_time() -> None:
    from vela.run.config import RunConfig

    config = RunConfig("XYZ", "2020-01-02", "2020-12-31", iv_rank=0.5)
    a, b = config.strategy(), config.strategy()
    assert a is not b and a.gate is not b.gate


# -- the runs root ------------------------------------------------------------


def test_runs_land_in_a_fixed_root_not_the_cwd(monkeypatch, tmp_path) -> None:
    """pictor reads one folder; it cannot follow a result to wherever the
    runner happened to stand."""
    from vela.run.artifacts import runs_root
    from vela.run.cli import parse

    base = ["XYZ", "--start", "2020-01-02", "--end", "2020-12-31"]
    monkeypatch.delenv("VELA_RUNS")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert runs_root() == tmp_path / ".vela" / "runs"

    monkeypatch.setenv("VELA_RUNS", "~/elsewhere")
    assert runs_root() == tmp_path / "elsewhere"
    assert parse(base).out == str(tmp_path / "elsewhere")  # recorded resolved
    assert parse([*base, "--out", "mine"]).out == "mine"  # --out still wins


# -- exit codes ---------------------------------------------------------------


def test_data_failures_are_told_apart() -> None:
    """What a driver does next differs for each, so each has its own code."""
    from antlia.history.errors import IngestFailed, NotCovered, NotEntitled

    from vela.data.guards import CoverageIncomplete
    from vela.data.market import data_failure

    assert data_failure(NotCovered("option_eod", "XYZ", [], "thetadata")) == "missing"
    assert data_failure(CoverageIncomplete("stale listing")) == "missing"
    assert data_failure(NotEntitled("thetadata", "before history begins")) == "denied"
    assert data_failure(IngestFailed("thetadata", "malformed rows")) == "fetch"
    assert data_failure(ValueError("a bug")) is None


@pytest.mark.parametrize(
    ("raised", "code"),
    [("missing", 3), ("refused", 4), ("denied", 5), ("fetch", 6)],
)
def test_each_failure_has_its_own_exit_code(monkeypatch, raised, code) -> None:
    from antlia.history.errors import IngestFailed, NotCovered, NotEntitled

    from vela.data.guards import CorporateActionSuspected
    from vela.run import config
    from vela.run.cli import main

    exc = {
        "missing": NotCovered("option_eod", "XYZ", [], "thetadata"),
        "refused": CorporateActionSuspected("a 10:1 split"),
        "denied": NotEntitled("thetadata", "before history begins"),
        "fetch": IngestFailed("thetadata", "malformed rows"),
    }[raised]

    def load(self):
        raise exc

    monkeypatch.setattr(config.RunConfig, "load", load)
    assert main(["XYZ", "--start", "2020-01-02", "--end", "2020-12-31"]) == code


def test_a_bug_in_the_read_path_crashes_rather_than_blaming_the_cache(monkeypatch) -> None:
    """Previously every load exception printed "warm the cache" and exited 1,
    indistinguishable from a cold store."""
    from vela.run import config
    from vela.run.cli import main

    def load(self):
        raise ZeroDivisionError("a bug")

    monkeypatch.setattr(config.RunConfig, "load", load)
    with pytest.raises(ZeroDivisionError):
        main(["XYZ", "--start", "2020-01-02", "--end", "2020-12-31"])
