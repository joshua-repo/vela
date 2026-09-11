# vela

A backtesting engine for option strategies. Daily bars, multi-leg positions,
concurrent holdings on one shared cash pool.

Vela **never talks to a data vendor.** Market data comes from
[antlia](https://github.com/joshua-repo/antlia), which is the single source of
truth shared with every other consumer -- so a backtest, a screen and a
dashboard cannot quietly disagree about what happened. Rendering belongs to
pictor. What lives here is the engine, the strategies, the risk models and the
metrics.

## What it is careful about

Most of the design exists because a wrong backtest looks exactly like a right
one. The number is plausible either way.

- **Lookahead is prevented, not audited.** A strategy sees the world only
  through a `Context` whose market is pinned to the current session; asking
  for a later one raises rather than answering.
- **Execution is attributed, not assumed away.** Every trade reports the
  spread it crossed and the commission it paid beside its gross P&L. On a
  defined-risk options strategy those are routinely the same order of
  magnitude as the result, and a net figure alone cannot tell a signal problem
  from an execution assumption.
- **A gated result is always shown against its own null.** A signal that does
  not beat always-entering has demonstrated nothing.
- **Small samples read as small.** Expectancy carries a standard error and a
  t statistic; win rate carries a Wilson interval. A run with too few trades
  says so instead of reporting a mean.
- **Refusals over plausible wrong answers.** An unadjusted price series that
  crosses a split, a window the data store cannot vouch for, a leg nobody is
  quoting: each stops the run and says why.
- **Every run records what it read** -- which append of the store, and whether
  that reading was verifiable -- so a result stays reproducible after the
  vendor restates.

Bounds are preferred to point estimates where the data cannot support one. A
stop is path-dependent and a daily bar has no path, so the engine can run the
same strategy as an envelope: close-only marking on one side, a re-ask at each
session's low on the other. The width between them is the finding.

## Install

Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q
```

## Run

```bash
uv run vela SYMBOL --start 2020-01-01 --end 2020-12-31
uv run vela SYMBOL --start ... --end ... --iv-rank 0.5 --envelope --breakeven
uv run vela --help
```

Runs are **offline by default**: a backtest never spends metered vendor
requests as a side effect. A symbol whose data is not cached stops and prints
the `antlia-history` commands that would warm it.

Each run writes `trades.csv`, `equity.csv` and a `summary.json` carrying the
full configuration, the data version and the result. The summary is the layer
anything downstream should read.

## Correctness

There is no reference implementation to check against, so the bar is
properties rather than a remembered number, and all of them are automated:

1. hand-computed reconciliation of cash, credit and P&L
2. the cash invariant, asserted every session
3. the expiration payoff identity, property-tested against an independently
   written payoff
4. byte-identical reproducibility

The suite is mutation-tested. Several of its early tests passed against a
deliberately broken engine and had to be rewritten -- a test that asserts
nothing happened usually asserts nothing at all.

## Status

Single-symbol backtesting of vertical spreads is complete: data guards,
portfolio book, Reg-T defined-risk margin, risk-based sizing, early
assignment, Black-Scholes implied volatility and greeks, an IV-rank entry
gate, per-trade and curve metrics, and run provenance.

Not built yet: parameter sweeps with cross-run aggregation, multi-symbol
books, naked-option margin, a volatility surface.
