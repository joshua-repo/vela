"""The data layer: antlia in, columnar snapshots out.

`snapshot` holds the shapes and knows nothing about antlia -- which is what
lets the engine be tested on hand-built chains. `market` is the only module
that imports antlia, and `guards` holds the judgements antlia deliberately
refuses to make.
"""

from vela.data.guards import (
    DIVERGENCE,
    CorporateActionSuspected,
    CoverageIncomplete,
    GuardFinding,
    check_corporate_actions,
    quote_divergence,
)
from vela.data.market import DataVersion, FutureData, MarketView, PinnedMarket
from vela.data.snapshot import ChainSnapshot, Quote, SpotBar

__all__ = [
    "DIVERGENCE",
    "ChainSnapshot",
    "DataVersion",
    "FutureData",
    "MarketView",
    "PinnedMarket",
    "CorporateActionSuspected",
    "CoverageIncomplete",
    "GuardFinding",
    "Quote",
    "SpotBar",
    "check_corporate_actions",
    "quote_divergence",
]
