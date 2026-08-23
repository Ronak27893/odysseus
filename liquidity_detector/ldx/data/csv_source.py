"""Local file adapter: CRSP exports, Nasdaq Data Link dumps, Polygon flat files.

This is the recommended production path. Point it at a directory of CSV or
Parquet files whose names match the Panel frames; whatever is present is
loaded and the rest defaults to empty.

    bars.parquet              ticker,date,open,high,low,close,volume   (REQUIRED)
    securities.parquet        ticker,first_date,last_date[,ipo_date,delist_date,
                              delist_reason,cik]                        (REQUIRED)
    float_history.parquet     ticker,date,float_shares
    corporate_actions.parquet ticker,date,kind,ratio
    filings.parquet           ticker,date,form
    label_events.parquet      ticker,date,kind
    control_events.parquet    ticker,date,kind
    intraday.parquet          ticker,date,closing_auction_share,small_trade_share

Two requirements the loader cannot verify for you and the audit will:
`bars` must be SPLIT-ADJUSTED, and `securities` must include DELISTED tickers.
A CRSP extract satisfies both; a screener export of currently-listed names
satisfies neither.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import Panel

_FRAMES = ("bars", "securities", "float_history", "corporate_actions",
           "filings", "label_events", "control_events", "intraday")


def _read(directory: Path, stem: str) -> pd.DataFrame | None:
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv),
                        (".csv.gz", pd.read_csv)):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return reader(p)
    return None


class CsvSource:
    """PanelSource backed by a directory of CSV/Parquet files."""

    name = "csv"

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"not a directory: {self.directory}")

    def load(self, start: str | None = None, end: str | None = None,
             tickers: list[str] | None = None) -> Panel:
        frames: dict[str, pd.DataFrame] = {}
        for stem in _FRAMES:
            df = _read(self.directory, stem)
            if df is not None:
                frames[stem] = df
        if "bars" not in frames:
            raise FileNotFoundError(f"bars.parquet/.csv not found in {self.directory}")
        if "securities" not in frames:
            raise FileNotFoundError(f"securities.parquet/.csv not found in {self.directory}")

        bars = frames["bars"]
        bars["date"] = pd.to_datetime(bars["date"])
        if start:
            bars = bars[bars["date"] >= pd.Timestamp(start)]
        if end:
            bars = bars[bars["date"] <= pd.Timestamp(end)]
        if tickers:
            keep = set(tickers)
            bars = bars[bars["ticker"].isin(keep)]
        frames["bars"] = bars.reset_index(drop=True)

        kwargs = {k: v for k, v in frames.items() if k not in ("bars", "securities")}
        return Panel(bars=frames["bars"], securities=frames["securities"],
                     source_name=f"csv:{self.directory.name}", **kwargs)
