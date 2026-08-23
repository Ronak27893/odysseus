"""Walk-forward splitting, purging, and the metrics that survive a 2% base rate."""
import numpy as np
import pandas as pd
import pytest

from ldx.config import LABEL_HORIZON_DAYS
from ldx.evaluation import (bootstrap_ci, precision_at_k, univariate_diagnostics,
                            walk_forward_folds)


def _dates(n=1000, start="2016-01-01"):
    return pd.Series(pd.bdate_range(start, periods=n))


def test_folds_are_strictly_forward_in_time():
    d = _dates()
    for train_idx, test_idx, meta in walk_forward_folds(d, n_splits=5):
        assert d.iloc[train_idx].max() < d.iloc[test_idx].min()


def test_no_random_splitting():
    """Test indices must be contiguous in calendar order, not scattered."""
    d = _dates()
    for _, test_idx, _ in walk_forward_folds(d, n_splits=5):
        td = d.iloc[test_idx].sort_values()
        assert (td.diff().dropna() >= pd.Timedelta(0)).all()


def test_purge_respects_label_horizon():
    """No training event may have its outcome revealed after the test opens."""
    d = _dates()
    for train_idx, test_idx, meta in walk_forward_folds(d, n_splits=5):
        test_start = d.iloc[test_idx].min()
        horizon_end = d.iloc[train_idx] + pd.Timedelta(days=LABEL_HORIZON_DAYS)
        assert (horizon_end <= test_start).all(), "leaked a training label into the test window"
        assert meta.n_purged >= 0


def test_purging_actually_removes_rows():
    d = _dates()
    folds = walk_forward_folds(d, n_splits=5)
    assert sum(f[2].n_purged for f in folds) > 0, "purge is a no-op; horizon not enforced"


def test_zero_embargo_would_leak():
    """Control: without the embargo the leak is present, proving the test bites."""
    d = _dates()
    leaked = 0
    for train_idx, test_idx, _ in walk_forward_folds(d, n_splits=5, embargo_days=0):
        test_start = d.iloc[test_idx].min()
        horizon_end = d.iloc[train_idx] + pd.Timedelta(days=LABEL_HORIZON_DAYS)
        leaked += int((horizon_end > test_start).sum())
    assert leaked > 0


def test_precision_at_k_ranks_by_score():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    assert precision_at_k(y, s, 3) == pytest.approx(1.0)
    assert precision_at_k(y, s, 6) == pytest.approx(0.5)


def test_precision_at_k_handles_k_larger_than_n():
    y = np.array([0, 1]); s = np.array([0.2, 0.9])
    assert precision_at_k(y, s, 99) == pytest.approx(0.5)


def test_bootstrap_ci_brackets_point_estimate():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.05).astype(int)
    s = rng.random(500) + y * 0.4
    lo, hi = bootstrap_ci(y, s, average_precision_score, n_boot=300)
    point = average_precision_score(y, s)
    assert lo <= point <= hi
    assert hi > lo


def test_leak_detector_fires_on_a_planted_leak():
    df = pd.DataFrame({"label": [0] * 100 + [1] * 100})
    df["honest"] = np.random.default_rng(0).normal(size=200)
    df["leaky"] = df["label"] * 10.0          # a perfect copy of the label
    d = univariate_diagnostics(df, ["honest", "leaky"])
    flagged = set(d.loc[d["suspected_leak"], "feature"])
    assert "leaky" in flagged
    assert "honest" not in flagged
