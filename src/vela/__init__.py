"""Vela -- a backtesting engine for option and equity strategies.

Vela never talks to a vendor. Market data comes from `antlia`, which is the
single source of truth shared with every other consumer, so a backtest and a
dashboard cannot disagree about what happened.
"""

from vela.instrument import CALL, PUT, Contract

__all__ = ["CALL", "PUT", "Contract"]
