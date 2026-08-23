"""The two data failures that manufacture the signature we hunt for."""
import numpy as np
import pandas as pd
import pytest

from ldx.data.base import Panel
from ldx.data.integrity import IntegrityError, audit_panel, detect_unadjusted_splits


def _series(n=400, price=10.0, volume=100_000.0, ticker="AAA", start="2020-01-01"):
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "ticker": ticker, "date": dates,
        "open": price, "high": price * 1.01, "low": price * 0.99,
        "close": price, "volume": volume,
    })


def _panel(bars, last_date=None, **kw):
    tickers = bars["ticker"].unique()
    sec = pd.DataFrame({
        "ticker": tickers,
        "first_date": [bars[bars.ticker == t]["date"].min() for t in tickers],
        "last_date": [last_date or bars[bars.ticker == t]["date"].max() for t in tickers],
        "ipo_date": [bars[bars.ticker == t]["date"].min() for t in tickers],
    })
    return Panel(bars=bars, securities=sec, **kw)


def test_unadjusted_reverse_split_is_detected():
    """Price x10 with volume /10 is a reverse split, not a run-up."""
    bars = _series()
    half = len(bars) // 2
    bars.loc[half:, ["open", "high", "low", "close"]] *= 10.0
    bars.loc[half:, "volume"] /= 10.0
    flagged = detect_unadjusted_splits(bars, None)
    assert len(flagged) == 1
    assert flagged.iloc[0]["matched_ratio"] == 10
    assert bool(flagged.iloc[0]["is_reverse"]) is True


def test_declared_split_is_not_flagged():
    """A declared corporate action explains the jump."""
    bars = _series()
    half = len(bars) // 2
    bars.loc[half:, ["open", "high", "low", "close"]] *= 10.0
    bars.loc[half:, "volume"] /= 10.0
    actions = pd.DataFrame([{"ticker": "AAA", "date": bars["date"].iloc[half],
                             "kind": "reverse_split", "ratio": 10.0}])
    assert len(detect_unadjusted_splits(bars, actions)) == 0


def test_genuine_ramp_is_not_a_split():
    """Price x10 with volume UP is a ramp. Volume must move inversely."""
    bars = _series()
    half = len(bars) // 2
    bars.loc[half:, ["open", "high", "low", "close"]] *= 10.0
    bars.loc[half:, "volume"] *= 50.0
    assert len(detect_unadjusted_splits(bars, None)) == 0


def test_adjusted_series_is_clean():
    rng = np.random.default_rng(0)
    bars = _series()
    walk = np.exp(np.cumsum(rng.normal(0, 0.02, len(bars))))
    for c in ("open", "high", "low", "close"):
        bars[c] = bars[c].to_numpy() * walk
    assert len(detect_unadjusted_splits(bars, None)) == 0


def test_survivor_only_universe_is_fatal():
    """A panel with no terminated securities has the positive class removed."""
    panel = _panel(_series())
    with pytest.raises(IntegrityError, match="terminated securities"):
        audit_panel(panel, strict=True)


def test_dead_tickers_pass_survivorship():
    bars = pd.concat([_series(ticker=f"T{i}") for i in range(20)], ignore_index=True)
    sec = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(20)],
        "first_date": bars["date"].min(),
        # Four of twenty terminate early -> 20% dead, above the 5% floor.
        "last_date": [bars["date"].max() if i >= 4 else bars["date"].iloc[100]
                      for i in range(20)],
        "ipo_date": bars["date"].min(),
    })
    rep = audit_panel(Panel(bars=bars, securities=sec), strict=False)
    assert rep.n_dead_tickers == 4
    assert not any("terminated securities" in f for f in rep.fatal)


def test_float_snapshot_is_warned_not_silently_used():
    bars = pd.concat([_series(ticker=f"T{i}") for i in range(20)], ignore_index=True)
    sec = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(20)],
        "first_date": bars["date"].min(),
        "last_date": [bars["date"].max() if i >= 4 else bars["date"].iloc[100]
                      for i in range(20)],
        "ipo_date": bars["date"].min(),
    })
    floats = pd.DataFrame([{"ticker": f"T{i}", "date": bars["date"].min(),
                            "float_shares": 1e7} for i in range(20)])
    rep = audit_panel(Panel(bars=bars, securities=sec, float_history=floats), strict=False)
    assert rep.float_history_is_snapshot
    assert any("snapshot" in w for w in rep.warnings)
