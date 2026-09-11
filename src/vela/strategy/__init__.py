"""Strategies: one method, one decision per day."""

from vela.strategy.base import Strategy
from vela.strategy.gate import AlwaysOpen, Gate, IVRank
from vela.strategy.vertical import VerticalSpread

__all__ = ["AlwaysOpen", "Gate", "IVRank", "Strategy", "VerticalSpread"]
