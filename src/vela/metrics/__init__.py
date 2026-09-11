"""What a run amounted to.

Two families, deliberately separate. `trades` measures the **decisions** --
one row per position, so a strategy with twelve trades reads as uncertain
because it is. `curve` measures the **account** -- one row per session,
which is what a Sharpe ratio is defined on.

Reporting only the second is how a strategy with no demonstrable edge gets
presented as a result: an equity curve always has a slope.

Implemented here rather than delegated. `quantstats` and friends own the
curve metrics well, but they arrive with matplotlib and yfinance attached,
and rendering belongs to pictor rather than to the backtest engine. The
formulas below are short, and being able to read them matters more than
importing them.
"""

from vela.metrics.curve import CurveStats, curve_stats
from vela.metrics.trades import TradeStats, trade_stats, wilson

__all__ = ["CurveStats", "TradeStats", "curve_stats", "trade_stats", "wilson"]
