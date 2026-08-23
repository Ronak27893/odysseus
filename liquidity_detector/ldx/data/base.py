"""Panel schema and the source contract every data adapter must satisfy.

The spec's first constraint is that data be point-in-time and include
delisted securities. That cannot be enforced by a comment, so it is enforced
here: a Panel validates its own shape, and `audit_panel` (integrity.py)
refuses a universe with no terminated securities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

REQUIRED_BAR_COLUMNS = ("ticker", "date", "open", "high", "low", "close", "volume")
REQUIRED_SECURITY_COLUMNS = ("ticker", "first_date", "last_date")

#: Outcome-label event kinds (the positive class).
LABEL_KINDS = ("sec_suspension", "delisting", "enforcement")

#: Confound kinds. These produce the same price/volume shape as a manufactured
#: event and are the controls the model must be scored against separately.
CONTROL_KINDS = (
    "biotech_readout",
    "ma_announcement",
    "index_inclusion",
    "short_squeeze",
    "earnings_surprise",
    "meme_attention",
)


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


@dataclass
class Panel:
    """A point-in-time market panel plus the regulatory record joined to it.

    Attributes
    ----------
    bars:
        Daily OHLCV, one row per (ticker, date). Prices MUST already be
        split-adjusted; `audit_panel` verifies this rather than trusting it.
    securities:
        One row per ticker with listing lifetime. `last_date` short of the
        panel end means the security terminated -- this is what makes the
        panel survivorship-free.
    float_history:
        As-of float records (step function). Float is read at event date, not
        from a current snapshot, because a vendor's `floatShares` today is the
        wrong number for an event three years ago.
    corporate_actions:
        Splits and reverse splits with ratios; used to distinguish a real
        reverse split from a suspected unadjusted price series.
    filings:
        EDGAR filing dates by form (424B5, S-3, S-1, 8-K, 25-NSE).
    label_events / control_events:
        Positive-class outcomes and labelled confounds respectively.
    intraday:
        Optional per (ticker, date) microstructure aggregates. Absent for most
        vendors; features derived from it are NaN rather than imputed.
    """

    bars: pd.DataFrame
    securities: pd.DataFrame
    float_history: pd.DataFrame = field(default_factory=lambda: _empty(("ticker", "date", "float_shares")))
    corporate_actions: pd.DataFrame = field(default_factory=lambda: _empty(("ticker", "date", "kind", "ratio")))
    filings: pd.DataFrame = field(default_factory=lambda: _empty(("ticker", "date", "form")))
    label_events: pd.DataFrame = field(default_factory=lambda: _empty(("ticker", "date", "kind")))
    control_events: pd.DataFrame = field(default_factory=lambda: _empty(("ticker", "date", "kind")))
    intraday: pd.DataFrame | None = None
    #: Free-form provenance, written into the integrity report.
    source_name: str = "unknown"

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED_BAR_COLUMNS if c not in self.bars.columns]
        if missing:
            raise ValueError(f"bars is missing required columns: {missing}")
        missing = [c for c in REQUIRED_SECURITY_COLUMNS if c not in self.securities.columns]
        if missing:
            raise ValueError(f"securities is missing required columns: {missing}")
        for frame_name in ("bars", "float_history", "corporate_actions", "filings",
                           "label_events", "control_events"):
            frame = getattr(self, frame_name)
            if "date" in frame.columns and len(frame):
                frame["date"] = pd.to_datetime(frame["date"])
        for col in ("first_date", "last_date", "ipo_date"):
            if col in self.securities.columns and len(self.securities):
                self.securities[col] = pd.to_datetime(self.securities[col])
        self.bars = self.bars.sort_values(["ticker", "date"]).reset_index(drop=True)

    @property
    def tickers(self) -> list[str]:
        return sorted(self.bars["ticker"].unique().tolist())

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.bars["date"].min(), self.bars["date"].max()

    def dead_tickers(self) -> set[str]:
        """Securities whose listing ended before the panel's last date."""
        if not len(self.securities):
            return set()
        panel_end = self.bars["date"].max()
        ended = self.securities["last_date"] < panel_end
        return set(self.securities.loc[ended, "ticker"])

    def float_as_of(self, ticker: str, when: pd.Timestamp) -> float | None:
        """Most recent float record at or before `when`. None if unknown."""
        if not len(self.float_history):
            return None
        rows = self.float_history
        rows = rows[(rows["ticker"] == ticker) & (rows["date"] <= when)]
        if not len(rows):
            return None
        return float(rows.sort_values("date")["float_shares"].iloc[-1])


class PanelSource(Protocol):
    """Adapter interface. Implementations live alongside this module."""

    name: str

    def load(self, start: str, end: str, tickers: list[str] | None = None) -> Panel:
        """Return a Panel covering [start, end]."""
        ...
