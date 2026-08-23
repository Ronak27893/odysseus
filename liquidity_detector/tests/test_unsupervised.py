"""The unsupervised screen: normalisation, weight renormalisation, ranking."""
import numpy as np
import pandas as pd
import pytest

from ldx.unsupervised import DEFAULT_WEIGHTS, rank, robust_z, score_events


def _events(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "event_date": pd.bdate_range("2022-01-03", periods=n),
        "float_turnover": rng.lognormal(0, 1, n),
        "concentration_ratio": rng.lognormal(2, 0.5, n),
        "peak_relative_volume": rng.lognormal(2, 1, n),
        "runup": rng.lognormal(0.2, 0.4, n),
        "prior_reverse_splits": rng.poisson(0.2, n),
        "offering_prior_10d": rng.integers(0, 2, n),
        "news_filing_in_window": rng.integers(0, 2, n),
        "overnight_share": rng.uniform(0, 1, n),
        "retracement": rng.uniform(-0.5, 1.2, n),
        "volume_decay": rng.lognormal(-0.5, 0.6, n),
    })


def test_robust_z_is_outlier_resistant():
    """One 124x print must not swamp the axis, as a mean/sd z-score would."""
    x = pd.Series([1.0] * 99 + [1000.0])
    z = robust_z(x)
    assert abs(z.iloc[0]) < 1.0
    assert z.iloc[-1] == 6.0          # clipped, not unbounded


def test_robust_z_handles_zero_spread():
    z = robust_z(pd.Series([5.0] * 50))
    assert (z == 0).all()


def test_score_is_finite_and_ordered():
    s = score_events(_events())
    assert np.isfinite(s["signature_score"]).all()
    assert s["signature_score"].std() > 0


def test_missing_features_renormalise_weights():
    """A source without EDGAR must still produce a coherent score."""
    df = _events().drop(columns=["offering_prior_10d", "news_filing_in_window"])
    s = score_events(df)
    assert set(s.attrs["features_missing"]) == {"offering_prior_10d", "news_filing_in_window"}
    assert "float_turnover" in s.attrs["features_used"]
    assert np.isfinite(s["signature_score"]).all()


def test_score_stays_bounded_after_renormalisation():
    """Renormalising over fewer axes must not inflate the scale."""
    full = score_events(_events())["signature_score"]
    partial = score_events(_events().drop(columns=["offering_prior_10d"]))["signature_score"]
    assert partial.abs().max() < full.abs().max() * 3


def test_direction_float_turnover_pushes_score_up():
    df = _events()
    df.loc[0, "float_turnover"] = df["float_turnover"].max() * 50
    s = score_events(df)
    assert s.loc[0, "z_float_turnover"] > 0


def test_direction_volume_decay_is_inverted():
    """A market that does NOT disappear is less suspicious, not more."""
    df = _events()
    df.loc[0, "volume_decay"] = 1e-4        # volume vanished
    df.loc[1, "volume_decay"] = 50.0        # volume grew
    s = score_events(df)
    assert s.loc[0, "z_volume_decay"] > s.loc[1, "z_volume_decay"]


def test_drivers_name_only_positive_contributors():
    s = score_events(_events())
    for drivers in s["drivers"]:
        assert "nan" not in str(drivers)
    assert (s["drivers"].str.len() > 0).any()


def test_rank_is_sorted_and_capped():
    out = rank(_events(), top_n=10)
    assert len(out) == 10
    assert out["signature_score"].is_monotonic_decreasing
    assert list(out["rank"]) == list(range(1, 11))


def test_rank_respects_asof_and_lookback():
    df = _events(n=300)
    asof = df["event_date"].iloc[200]
    out = rank(df, top_n=50, asof=asof, lookback_days=30)
    assert (out["event_date"] <= asof).all()
    assert (out["event_date"] >= asof - pd.Timedelta(days=30)).all()


def test_no_usable_features_raises():
    with pytest.raises(ValueError, match="no usable features"):
        score_events(pd.DataFrame({"ticker": ["A"] * 50,
                                   "event_date": pd.bdate_range("2022-01-03", periods=50),
                                   "irrelevant": range(50)}))


def test_weights_follow_the_hypothesis():
    """float_turnover carries the most weight: the claim is about exit depth."""
    w = {name: weight for name, weight, *_ in DEFAULT_WEIGHTS}
    assert w["float_turnover"] == max(w.values())
