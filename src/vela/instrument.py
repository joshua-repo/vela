"""What a tradable thing is, and what makes two of them the same.

The option key is deliberately **antlia's own natural key** --
`(symbol, expiration, strike, right)`, the OCC identity. Sharing it means a
contract needs no translation on the way in and no id map to exist first:
a row from `history.option_eod` names a `Contract` directly.

Conventions, pinned here so nothing downstream has to guess:

- `strike` is in **dollars**, never thousandths.
- `right` is `"C"` or `"P"`, one letter, matching antlia and OCC.
- `symbol` is the **underlying root**, never a contract symbol: every option
  row for one underlying carries the same `symbol`, and the contract is the
  whole key.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

#: Option rights. One letter, as antlia stores them.
CALL = "C"
PUT = "P"

type Right = Literal["C", "P"]

#: Contracts per option contract. Equity options are 100 in every case Vela
#: currently serves; adjusted contracts (post-split, non-standard deliverable)
#: are exactly the case `guards.CorporateActionGuard` refuses to run on, so
#: this constant is safe *because* that guard exists.
MULTIPLIER = 100


def as_right(value: str) -> Right:
    """`"C"`/`"P"` from whatever spelling a caller had to hand."""
    text = value.strip().upper()
    if text in {"C", "CALL"}:
        return "C"
    if text in {"P", "PUT"}:
        return "P"
    raise ValueError(f"unknown right {value!r}; use 'C'/'call' or 'P'/'put'")


@dataclass(frozen=True, slots=True, order=True)
class Contract:
    """One option contract. Frozen and hashable -- it is used as a book key."""

    symbol: str
    expiration: dt.date
    strike: float
    right: Right

    def dte(self, on: dt.date) -> int:
        """Calendar days to expiration, date to date.

        Date-to-date deliberately: deriving DTE from the vendor's 17:15
        timestamp floors a day short, which biases every downstream IV
        slightly high. hawk shipped that bug for months.
        """
        return (self.expiration - on).days

    def intrinsic(self, spot: float) -> float:
        """Per-share exercise value at `spot`, never negative."""
        if self.right == "P":
            return max(self.strike - spot, 0.0)
        return max(spot - self.strike, 0.0)

    def __str__(self) -> str:
        return f"{self.symbol} {self.expiration:%Y-%m-%d} {self.strike:g}{self.right}"
