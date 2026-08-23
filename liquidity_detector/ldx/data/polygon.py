"""Polygon.io adapter.

Chosen because its reference endpoint exposes DELISTED tickers, which is the
single thing that makes a survivorship-free universe reachable without a CRSP
licence. The `active=false` call below is not an optional nicety -- omitting it
reproduces exactly the failure that made the earlier exploratory pass
unusable, where suspended issuers had no price history to fetch.

Requires POLYGON_API_KEY. Not exercised against the live API in environments
without egress to polygon.io; the request shapes follow the v2/v3 REST spec.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

from .base import Panel

_BASE = "https://api.polygon.io"


class PolygonSource:
    """PanelSource over Polygon REST, with on-disk caching."""

    name = "polygon"

    def __init__(self, api_key: str | None = None, cache_dir: str | Path = ".cache/polygon",
                 max_retries: int = 4, pause: float = 0.25):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise RuntimeError("POLYGON_API_KEY is not set")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.pause = pause
        self._session = requests.Session()

    # -- transport ---------------------------------------------------------
    def _get(self, url: str, params: dict | None = None) -> dict:
        params = dict(params or {})
        params["apiKey"] = self.api_key
        delay = 1.0
        for attempt in range(self.max_retries):
            r = self._session.get(url, params=params, timeout=45)
            if r.status_code == 200:
                time.sleep(self.pause)
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
        raise RuntimeError(f"polygon request failed after {self.max_retries} attempts: {url}")

    def _paginate(self, url: str, params: dict | None = None) -> list[dict]:
        out: list[dict] = []
        payload = self._get(url, params)
        out.extend(payload.get("results", []) or [])
        while payload.get("next_url"):
            payload = self._get(payload["next_url"])
            out.extend(payload.get("results", []) or [])
        return out

    # -- universe ----------------------------------------------------------
    def list_securities(self, include_delisted: bool = True) -> pd.DataFrame:
        """Reference universe. `include_delisted` is what defeats survivorship."""
        rows = self._paginate(f"{_BASE}/v3/reference/tickers",
                              {"market": "stocks", "active": "true", "limit": 1000})
        if include_delisted:
            rows += self._paginate(f"{_BASE}/v3/reference/tickers",
                                   {"market": "stocks", "active": "false", "limit": 1000})
        df = pd.DataFrame(rows)
        if not len(df):
            return pd.DataFrame(columns=["ticker", "first_date", "last_date"])
        out = pd.DataFrame({
            "ticker": df["ticker"],
            "cik": df.get("cik"),
            "first_date": pd.to_datetime(df.get("list_date"), errors="coerce"),
            "last_date": pd.to_datetime(df.get("delisted_utc"), errors="coerce"),
            "ipo_date": pd.to_datetime(df.get("list_date"), errors="coerce"),
            "delist_date": pd.to_datetime(df.get("delisted_utc"), errors="coerce"),
            "active": df.get("active"),
        })
        return out.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)

    # -- bars --------------------------------------------------------------
    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Split-adjusted daily bars. `adjusted=true` is verified by the audit."""
        cache_file = self.cache / f"{ticker}_{start}_{end}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        url = f"{_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        rows = self._paginate(url, {"adjusted": "true", "sort": "asc", "limit": 50000})
        if not rows:
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low",
                                         "close", "volume"])
        df = pd.DataFrame(rows)
        out = pd.DataFrame({
            "ticker": ticker,
            "date": pd.to_datetime(df["t"], unit="ms").dt.normalize(),
            "open": df["o"].astype(float), "high": df["h"].astype(float),
            "low": df["l"].astype(float), "close": df["c"].astype(float),
            "volume": df["v"].astype(float),
        })
        out.to_parquet(cache_file, index=False)
        return out

    def load(self, start: str, end: str, tickers: list[str] | None = None) -> Panel:
        sec = self.list_securities(include_delisted=True)
        if tickers:
            sec = sec[sec["ticker"].isin(set(tickers))]
        parts = []
        for tk in sec["ticker"]:
            try:
                b = self.daily_bars(tk, start, end)
            except Exception:
                continue
            if len(b):
                parts.append(b)
        if not parts:
            raise RuntimeError("polygon returned no bars for the requested universe")
        bars = pd.concat(parts, ignore_index=True)
        obs = bars.groupby("ticker")["date"].agg(["min", "max"]).rename(
            columns={"min": "obs_first", "max": "obs_last"})
        sec = sec.merge(obs, left_on="ticker", right_index=True, how="right")
        sec["first_date"] = sec["first_date"].fillna(sec["obs_first"])
        sec["last_date"] = sec["last_date"].fillna(sec["obs_last"])
        return Panel(bars=bars, securities=sec.drop(columns=["obs_first", "obs_last"]),
                     source_name="polygon")
