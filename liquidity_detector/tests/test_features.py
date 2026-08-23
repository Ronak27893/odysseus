"""Feature arithmetic, checked against hand-computable series."""
import numpy as np
import pandas as pd
import pytest

from ldx.config import PRIMARY_WINDOW
from ldx.data.base import Panel
from ldx.features import build_features, detect_candidates

N_BASE = 300
W = PRIMARY_WINDOW


def _build(ramp_closes, ramp_volume=200_000.0, base_vol=1_000.0, base_px=10.0,
           float_shares=500_000.0, overnight="none", post_vol=100.0, n_post=40):
    """Baseline -> 5-day event -> quiet tail. `overnight` controls the gap split."""
    n = N_BASE + len(ramp_closes) + n_post
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = np.concatenate([
        np.full(N_BASE, base_px), np.asarray(ramp_closes, dtype=float),
        np.full(n_post, float(ramp_closes[-1]) * 0.2),
    ])
    volume = np.concatenate([
        np.full(N_BASE, base_vol), np.full(len(ramp_closes), ramp_volume),
        np.full(n_post, post_vol),
    ])
    open_ = np.empty(n)
    open_[0] = close[0]
    if overnight == "all":          # gap to today's close, then flat intraday
        open_[1:] = close[1:]
    elif overnight == "none":       # open at yesterday's close, move intraday
        open_[1:] = close[:-1]
    else:
        open_[1:] = np.sqrt(close[1:] * close[:-1])
    bars = pd.DataFrame({
        "ticker": "AAA", "date": dates, "open": open_,
        "high": np.maximum(open_, close), "low": np.minimum(open_, close),
        "close": close, "volume": volume,
    })
    sec = pd.DataFrame([{"ticker": "AAA", "first_date": dates[0], "last_date": dates[-1],
                         "ipo_date": dates[0]}])
    floats = pd.DataFrame([{"ticker": "AAA", "date": dates[0], "float_shares": float_shares}])
    return Panel(bars=bars, securities=sec, float_history=floats), bars


def _event(panel, window=W):
    f = build_features(panel, windows=(window,))
    assert len(f) >= 1, "no candidate event detected"
    return f.sort_values("liquidity_concentration").iloc[-1]


def test_float_turnover_is_window_volume_over_float():
    panel, _ = _build([12, 18, 26, 36, 50], ramp_volume=200_000.0, float_shares=500_000.0)
    row = _event(panel)
    # 5 days x 200k shares = 1.0M shares against a 500k float.
    assert row["float_turnover"] == pytest.approx(1_000_000 / 500_000, rel=1e-9)


def test_liquidity_concentration_matches_definition(  ):
    panel, bars = _build([12, 18, 26, 36, 50])
    row = _event(panel)
    dv = (bars["close"] * bars["volume"]).to_numpy()
    end = int(np.flatnonzero(bars["date"].to_numpy() == np.datetime64(row["event_date"]))[0])
    win = dv[end - W + 1: end + 1].sum()
    trail = dv[max(0, end - 504 + 1): end + 1].sum()
    assert row["liquidity_concentration"] == pytest.approx(win / trail, rel=1e-9)


def test_runup_is_peak_over_pre_window_median():
    panel, _ = _build([12, 18, 26, 36, 50], base_px=10.0)
    row = _event(panel)
    assert row["runup"] == pytest.approx(50.0 / 10.0, rel=1e-9)


def test_retracement_full_round_trip_is_one():
    """Peak 50, baseline 10, settles back to 10 -> retracement 1.0."""
    panel, _ = _build([12, 18, 26, 36, 50])
    # tail settles at 50*0.2 = 10.0, exactly the pre-window baseline
    row = _event(panel)
    assert row["retracement"] == pytest.approx(1.0, abs=1e-9)


def test_volume_decay_is_post_over_pre():
    panel, _ = _build([12, 18, 26, 36, 50], base_vol=1_000.0, post_vol=100.0)
    row = _event(panel)
    assert row["volume_decay"] == pytest.approx(100.0 / 1_000.0, rel=1e-9)


def test_overnight_share_all_gap():
    """Every move happens between the close and the next open -> share 1."""
    panel, _ = _build([12, 18, 26, 36, 50], overnight="all")
    row = _event(panel)
    assert row["overnight_share"] == pytest.approx(1.0, abs=1e-6)


def test_overnight_share_all_intraday():
    """Every move happens inside the session -> share 0."""
    panel, _ = _build([12, 18, 26, 36, 50], overnight="none")
    row = _event(panel)
    assert row["overnight_share"] == pytest.approx(0.0, abs=1e-6)


def test_overnight_share_half():
    panel, _ = _build([12, 18, 26, 36, 50], overnight="half")
    row = _event(panel)
    assert row["overnight_share"] == pytest.approx(0.5, abs=1e-6)


def test_retracement_undefined_without_real_runup():
    """No run-up means the denominator is noise; the feature must be NaN."""
    panel, _ = _build([10.01, 10.0, 10.02, 9.99, 10.0], ramp_volume=500_000.0)
    row = _event(panel)
    assert np.isnan(row["retracement"])


def test_candidate_refractory_collapses_one_episode():
    """A single episode yields one anchor, not a smear of overlapping windows."""
    _, bars = _build([12, 18, 26, 36, 50])
    anchors = detect_candidates(bars["close"].to_numpy(), bars["volume"].to_numpy())
    assert len(anchors) == 1


def test_windows_share_one_anchor():
    """3d/5d/10d are measured on the SAME event, not three event sets."""
    panel, _ = _build([12, 18, 26, 36, 50])
    f = build_features(panel, windows=(3, 5, 10))
    assert f["event_date"].nunique() == 1
    assert sorted(f["window"].tolist()) == [3, 5, 10]
