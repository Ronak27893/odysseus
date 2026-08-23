"""yfinance adapter.

Usable and free, with one structural limitation that governs everything else:
**yfinance is survivor-only**. Yahoo drops history for suspended and delisted
issuers, which is exactly the population the positive class is drawn from. So
a yfinance panel supports the *screen* (rank live names by signature) but not
supervised training or validation, because the positive class is unavailable
by construction. `audit_panel` will say so rather than let it pass quietly.

Three choices here address failures the earlier exploratory pass hit:

* `auto_adjust=False` and the **`Close`** column. Yahoo's `Close` is
  split-adjusted but not dividend-adjusted, and `Volume` is split-adjusted to
  match. That is precisely what this model needs: dividend adjustment would
  distort both the traded price level and dollar volume. `Adj Close` is
  deliberately not used.
* Share counts come from `get_shares_full()`, a **time series**, not from the
  `floatShares` snapshot in `.info`. A snapshot is the wrong number for an
  event three years ago and is the most likely cause of implausible turnover
  readings such as 124x.
* Share counts are recorded as `shares_outstanding`, not float. Float is
  smaller, so turnover computed against it is a **lower bound** on true float
  turnover. The adapter does not silently pass one off as the other.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Panel

#: Yahoo rejects very large batches; chunk and pause between them.
_CHUNK = 40
_PAUSE = 1.0


class YFinanceSource:
    """PanelSource over yfinance, with on-disk caching.

    Parameters
    ----------
    tickers:
        Universe to fetch. Supply delisted names here if you have them from
        another source; yfinance usually will not return history for them, and
        those that come back empty are reported rather than dropped silently.
    use_shares_history:
        Fetch `get_shares_full()` per ticker. Slower (one request each) but it
        is the difference between a point-in-time denominator and a snapshot.
    """

    name = "yfinance"

    def __init__(self, tickers: list[str], start: str = "2015-01-01",
                 end: str | None = None, cache_dir: str | Path = ".cache/yfinance",
                 use_shares_history: bool = True, max_retries: int = 3):
        if not tickers:
            raise ValueError("tickers must be a non-empty list")
        self.tickers = [t.strip().upper() for t in tickers if t and t.strip()]
        self.start = start
        self.end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.use_shares_history = use_shares_history
        self.max_retries = max_retries
        self.missing: list[str] = []

    # -- bars --------------------------------------------------------------
    def _download_chunk(self, chunk: list[str]):
        import yfinance as yf
        delay = 2.0
        for _ in range(self.max_retries):
            try:
                return yf.download(
                    chunk, start=self.start, end=self.end,
                    auto_adjust=False,          # keep split-adjusted Close, not Adj Close
                    actions=False, group_by="ticker", threads=True,
                    progress=False,
                )
            except Exception:
                time.sleep(delay)
                delay *= 2
        return None

    @staticmethod
    def _extract(raw, ticker: str) -> pd.DataFrame | None:
        """Pull one ticker's OHLCV out of yfinance's variable column shapes."""
        if raw is None or not len(raw):
            return None
        try:
            df = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            return None
        need = ("Open", "High", "Low", "Close", "Volume")
        if not all(c in df.columns for c in need):
            return None
        out = pd.DataFrame({
            "ticker": ticker,
            "date": pd.to_datetime(df.index).tz_localize(None).normalize(),
            "open": pd.to_numeric(df["Open"], errors="coerce"),
            "high": pd.to_numeric(df["High"], errors="coerce"),
            "low": pd.to_numeric(df["Low"], errors="coerce"),
            "close": pd.to_numeric(df["Close"], errors="coerce"),
            "volume": pd.to_numeric(df["Volume"], errors="coerce"),
        }).reset_index(drop=True)
        out = out.dropna(subset=["close", "volume"])
        out = out[(out["close"] > 0) & (out["volume"] >= 0)]
        return out if len(out) else None

    def load_bars(self) -> pd.DataFrame:
        cache_file = self.cache / f"bars_{self.start}_{self.end}_{len(self.tickers)}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        parts = []
        for i in range(0, len(self.tickers), _CHUNK):
            chunk = self.tickers[i:i + _CHUNK]
            raw = self._download_chunk(chunk)
            for tk in chunk:
                got = self._extract(raw, tk)
                if got is None:
                    self.missing.append(tk)
                else:
                    parts.append(got)
            time.sleep(_PAUSE)
        if not parts:
            raise RuntimeError(
                "yfinance returned no usable bars. If the universe is mostly "
                "suspended or delisted names, this is the expected outcome -- "
                "Yahoo drops their history, which is the survivorship problem.")
        bars = pd.concat(parts, ignore_index=True)
        bars.to_parquet(cache_file, index=False)
        return bars

    # -- corporate actions and share counts ---------------------------------
    def load_actions_and_shares(self, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        import yfinance as yf
        actions, shares = [], []
        for tk in tickers:
            try:
                t = yf.Ticker(tk)
            except Exception:
                continue
            try:
                sp = t.splits
                if sp is not None and len(sp):
                    for dt, ratio in sp.items():
                        ratio = float(ratio)
                        if ratio <= 0:
                            continue
                        actions.append({
                            "ticker": tk,
                            "date": pd.Timestamp(dt).tz_localize(None).normalize(),
                            # yfinance reports a 1-for-10 reverse split as 0.1.
                            "kind": "reverse_split" if ratio < 1 else "split",
                            "ratio": (1.0 / ratio) if ratio < 1 else ratio,
                        })
            except Exception:
                pass
            if self.use_shares_history:
                try:
                    sh = t.get_shares_full(start=self.start, end=self.end)
                    if sh is not None and len(sh):
                        sh = sh[~sh.index.duplicated(keep="last")]
                        for dt, n in sh.items():
                            if pd.notna(n) and float(n) > 0:
                                shares.append({
                                    "ticker": tk,
                                    "date": pd.Timestamp(dt).tz_localize(None).normalize(),
                                    "float_shares": float(n),
                                })
                except Exception:
                    pass
            time.sleep(0.15)
        return (pd.DataFrame(actions, columns=["ticker", "date", "kind", "ratio"]),
                pd.DataFrame(shares, columns=["ticker", "date", "float_shares"]))

    # -- assembly -----------------------------------------------------------
    def load(self, start: str | None = None, end: str | None = None,
             tickers: list[str] | None = None) -> Panel:
        bars = self.load_bars()
        present = sorted(bars["ticker"].unique().tolist())
        actions, shares = self.load_actions_and_shares(present)

        obs = bars.groupby("ticker")["date"].agg(["min", "max"])
        securities = pd.DataFrame({
            "ticker": obs.index,
            "first_date": obs["min"].to_numpy(),
            # Observed last bar. For a live name this is the panel end, which
            # is why a yfinance universe reads as ~0% terminated.
            "last_date": obs["max"].to_numpy(),
            "ipo_date": obs["min"].to_numpy(),
            "delist_date": pd.NaT,
            "delist_reason": "",
        }).reset_index(drop=True)

        panel = Panel(bars=bars, securities=securities, float_history=shares,
                      corporate_actions=actions, source_name="yfinance")
        # Recorded so downstream code and the audit can state the limitation
        # explicitly rather than inferring it.
        panel.survivorship_note = (
            "yfinance is survivor-only: Yahoo drops history for suspended and "
            "delisted issuers. Supervised training and validation are not "
            "supported on this source; use it for the unsupervised screen.")
        panel.missing_tickers = list(self.missing)
        panel.share_count_basis = "shares_outstanding"
        return panel


def load_universe_file(path: str | Path) -> list[str]:
    """Read a ticker list from a text file (one per line) or a CSV `ticker` column."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        col = next((c for c in df.columns if c.strip().lower() == "ticker"), df.columns[0])
        vals = df[col].astype(str).tolist()
    else:
        vals = [ln.strip() for ln in p.read_text().splitlines()]
    return [v.strip().upper() for v in vals if v.strip() and not v.startswith("#")]
