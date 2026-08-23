"""End-to-end run: panel -> integrity -> features -> labels -> models -> deliverables."""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .config import FEATURES, PATHS, PRIMARY_WINDOW, EVENT_WINDOWS
from .data.integrity import audit_panel, detect_unadjusted_splits
from .evaluation import (calibration_plot, evaluate_walk_forward, save_evaluation,
                         univariate_diagnostics)
from .features import build_features
from .labeling import attach_labels, censoring_report
from .model import build_gbm, build_logistic, logistic_coefficients
from .screen import build_screen


@dataclass
class RunResult:
    features: pd.DataFrame
    integrity: dict
    censoring: dict
    results: list
    coefficients: pd.DataFrame
    screen: pd.DataFrame
    robustness: pd.DataFrame
    diagnostics: pd.DataFrame


def run(panel, paths=PATHS, n_splits: int = 5, strict_integrity: bool = True,
        verbose: bool = True) -> RunResult:
    paths.ensure()
    log = print if verbose else (lambda *a, **k: None)

    # --- 1. integrity ------------------------------------------------------
    report = audit_panel(panel, strict=strict_integrity)
    report.save(paths.integrity)
    log(report.summary())

    # --- 2. features -------------------------------------------------------
    split_flags = detect_unadjusted_splits(panel.bars, panel.corporate_actions)
    feats = build_features(panel, windows=EVENT_WINDOWS, split_flags=split_flags)
    if not len(feats):
        raise RuntimeError("no candidate events found")
    feats = attach_labels(feats, panel)
    feats.to_parquet(paths.feature_store, index=False)
    log(f"\nfeature store -> {paths.feature_store}  {feats.shape}")

    primary = feats[(feats["window"] == PRIMARY_WINDOW) & feats["eligible"] & ~feats["censored"]]
    primary = primary.reset_index(drop=True)
    cens = censoring_report(feats[feats["window"] == PRIMARY_WINDOW])
    log(f"\nevents={cens['n_events']} positives={cens['n_positive']} "
        f"base_rate={cens['base_rate']:.2%} censored={cens['n_censored']} "
        f"controls={cens['n_control']}")
    log(f"modelling set after eligibility+censoring filters: {len(primary)} events, "
        f"{int(primary['label'].sum())} positive")

    # --- 2b. leak check before any model is fitted -------------------------
    diag_cols = FEATURES.model_columns("forensic", include_intraday=True) + list(FEATURES.diagnostics)
    diag_cols = [c for c in diag_cols if c in primary.columns]
    diagnostics = univariate_diagnostics(primary, diag_cols)
    log("\nunivariate feature diagnostics (leak check before fitting):")
    log(diagnostics.to_string(index=False))
    leaks = diagnostics[diagnostics["suspected_leak"]] if len(diagnostics) else diagnostics
    if len(leaks):
        log(f"\n!! {len(leaks)} feature(s) separate almost perfectly on their own: "
            f"{list(leaks['feature'])}. Treat as a suspected leak, not a result.")

    # --- 3. models: logistic first, then GBM; screen and forensic modes ----
    results, oofs = [], {}
    for mode in ("screen", "forensic"):
        cols = FEATURES.model_columns(mode, include_intraday=True)
        cols = [c for c in cols if primary[c].notna().any()]
        for name, factory in (("logistic", build_logistic), ("gbm", build_gbm)):
            res, oof = evaluate_walk_forward(primary, cols, factory, name, mode,
                                             n_splits=n_splits)
            results.append(res)
            oofs[(name, mode)] = oof
            log("\n" + res.summary())

    # --- 4. fit final models on all eligible data --------------------------
    screen_cols = [c for c in FEATURES.model_columns("screen", include_intraday=True)
                   if primary[c].notna().any()]
    forensic_cols = [c for c in FEATURES.model_columns("forensic", include_intraday=True)
                     if primary[c].notna().any()]
    final_screen = build_logistic(screen_cols)
    final_screen.fit(primary[screen_cols], primary["label"])
    final_forensic = build_logistic(forensic_cols)
    final_forensic.fit(primary[forensic_cols], primary["label"])
    joblib.dump({"screen": final_screen, "screen_columns": screen_cols,
                 "forensic": final_forensic, "forensic_columns": forensic_cols},
                paths.model)
    coefs = logistic_coefficients(final_forensic, forensic_cols).to_frame()
    log("\nlogistic coefficients (forensic, standardised):")
    log(coefs.to_string(index=False))

    # --- 5. calibration on the best-documented model -----------------------
    calibration_plot(oofs[("logistic", "forensic")], paths.calibration_plot,
                     title="Calibration - logistic / forensic (out-of-fold)")
    log(f"\ncalibration plot -> {paths.calibration_plot}")

    # --- 6. ranked screen ---------------------------------------------------
    scr = build_screen(primary, final_screen, screen_cols)
    scr.to_csv(paths.screen, index=False)
    log(f"ranked screen -> {paths.screen}  ({len(scr)} rows)")

    # --- 7. window robustness (3d / 5d / 10d on the SAME events) -----------
    rob_rows = []
    for W in EVENT_WINDOWS:
        sub = feats[(feats["window"] == W) & feats["eligible"] & ~feats["censored"]]
        sub = sub.reset_index(drop=True)
        if sub["label"].sum() < 3:
            continue
        cols = [c for c in FEATURES.model_columns("forensic", include_intraday=True)
                if sub[c].notna().any()]
        r, _ = evaluate_walk_forward(sub, cols, build_logistic, "logistic",
                                     f"forensic-{W}d", n_splits=n_splits)
        rob_rows.append({"window": W, "n_events": r.n_events, "n_positive": r.n_positive,
                         "pr_auc": r.pr_auc, "pr_auc_lo": r.pr_auc_ci[0],
                         "pr_auc_hi": r.pr_auc_ci[1], "lift": r.pr_auc_lift})
    robustness = pd.DataFrame(rob_rows)
    log("\nwindow robustness (same events, three measurement scales):")
    log(robustness.to_string(index=False))

    save_evaluation(results, paths.evaluation, extra={
        "integrity": report.to_dict(),
        "censoring": cens,
        "coefficients": coefs.to_dict(orient="records"),
        "robustness": robustness.to_dict(orient="records"),
        "univariate_diagnostics": diagnostics.to_dict(orient="records"),
        "source": getattr(panel, "source_name", "unknown"),
    })
    log(f"evaluation -> {paths.evaluation}")

    return RunResult(features=feats, integrity=report.to_dict(), censoring=cens,
                     results=results, coefficients=coefs, screen=scr,
                     robustness=robustness, diagnostics=diagnostics)
