"""End-to-end smoke test: every deliverable is produced and internally consistent."""
import json
from pathlib import Path

import pandas as pd
import pytest

from ldx.config import Paths
from ldx.data.integrity import audit_panel
from ldx.data.synthetic import SyntheticConfig, generate_panel
from ldx.pipeline import run


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    paths = Paths(root=tmp_path_factory.mktemp("artifacts"))
    panel = generate_panel(SyntheticConfig(n_tickers=320, seed=5))
    return run(panel, paths=paths, n_splits=4, verbose=False), paths


def test_all_deliverables_written(result):
    _, paths = result
    for p in (paths.feature_store, paths.model, paths.calibration_plot,
              paths.screen, paths.evaluation, paths.integrity):
        assert p.exists(), f"missing deliverable: {p.name}"
        assert p.stat().st_size > 0


def test_feature_store_roundtrips_as_parquet(result):
    _, paths = result
    df = pd.read_parquet(paths.feature_store)
    assert len(df) > 0
    assert {"ticker", "event_date", "window", "label"} <= set(df.columns)
    assert set(df["window"].unique()) == {3, 5, 10}


def test_synthetic_panel_passes_integrity(result):
    panel = generate_panel(SyntheticConfig(n_tickers=320, seed=5))
    rep = audit_panel(panel, strict=False)
    assert rep.ok, rep.summary()
    assert rep.dead_ticker_share > 0.05


def test_evaluation_reports_all_four_model_mode_combinations(result):
    res, paths = result
    payload = json.loads(paths.evaluation.read_text())
    combos = {(r["model"], r["mode"]) for r in payload["results"]}
    assert combos == {("logistic", "screen"), ("gbm", "screen"),
                      ("logistic", "forensic"), ("gbm", "forensic")}


def test_metrics_are_pr_based_and_carry_intervals(result):
    res, paths = result
    payload = json.loads(paths.evaluation.read_text())
    for r in payload["results"]:
        assert "pr_auc" in r and "pr_auc_ci" in r
        assert r["precision_at_k"], "precision@k missing"
        lo, hi = r["pr_auc_ci"]
        assert lo <= r["pr_auc"] <= hi


def test_control_false_positive_rate_is_reported(result):
    res, paths = result
    payload = json.loads(paths.evaluation.read_text())
    for r in payload["results"]:
        assert r["control_flag_rate"], "control FPR must be reported, not just random-negative FPR"


def test_screen_is_ranked_and_uses_causal_columns_only(result):
    res, paths = result
    scr = pd.read_csv(paths.screen)
    assert len(scr) > 0
    assert scr["score"].is_monotonic_decreasing
    for c in ("retracement", "volume_decay"):
        assert c not in scr.columns


def test_censoring_is_reported(result):
    res, _ = result
    assert "n_censored" in res.censoring
    assert "undetected manipulation" in res.censoring["note"].lower()


def test_window_robustness_covers_all_three_scales(result):
    res, _ = result
    assert set(res.robustness["window"]) == {3, 5, 10}
