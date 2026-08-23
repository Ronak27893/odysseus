"""Outcome labelling, control tagging, and censoring accounting.

The label is deliberately coarse: a regulatory or exchange TERMINATION within
`LABEL_HORIZON_DAYS` of the event. Two consequences are structural and are
recorded rather than smoothed over:

* Undetected manipulation sits in the negative class. Enforcement is a
  filtered, lagged, resource-constrained observation of manipulation, not a
  census of it. Reported precision is therefore a LOWER bound and reported
  recall is against detected cases only.
* Events too close to the panel end cannot have their outcome observed. Those
  rows are censored and excluded from training rather than labelled 0, which
  would teach the model that recent events are safe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LABEL_HORIZON_DAYS, MAX_MARKET_CAP, MIN_PRICE

#: Delisting reasons that are NOT adverse outcomes. In real EDGAR data a
#: Form 25-NSE does not distinguish these from deficiency delistings without
#: reading the filing; treating them as positives would poison the label.
BENIGN_DELIST_REASONS = ("merger", "going_private", "acquired", "")


def attach_labels(features: pd.DataFrame, panel,
                  horizon_days: int = LABEL_HORIZON_DAYS,
                  control_match_days: int = 10) -> pd.DataFrame:
    """Join outcomes, controls, eligibility and censoring onto the features."""
    if not len(features):
        return features
    df = features.copy()
    panel_end = pd.Timestamp(panel.bars["date"].max())

    # --- positive class ---------------------------------------------------
    labels = panel.label_events
    df["label"] = 0
    df["label_kind"] = ""
    df["days_to_label"] = np.nan
    if len(labels):
        by_ticker = {tk: gg.sort_values("date") for tk, gg in labels.groupby("ticker")}
        lab, kind, dtl = [], [], []
        for tk, ev in zip(df["ticker"].to_numpy(), df["event_date"].to_numpy()):
            gg = by_ticker.get(tk)
            if gg is None:
                lab.append(0); kind.append(""); dtl.append(np.nan); continue
            delta = (gg["date"] - pd.Timestamp(ev)).dt.days
            hit = gg[(delta >= 0) & (delta <= horizon_days)]
            if len(hit):
                lab.append(1)
                kind.append(str(hit.iloc[0]["kind"]))
                dtl.append(float((pd.Timestamp(hit.iloc[0]["date"]) - pd.Timestamp(ev)).days))
            else:
                lab.append(0); kind.append(""); dtl.append(np.nan)
        df["label"] = lab; df["label_kind"] = kind; df["days_to_label"] = dtl

    # --- confound controls -------------------------------------------------
    df["control_kind"] = ""
    controls = panel.control_events
    if len(controls):
        by_ticker = {tk: gg.sort_values("date") for tk, gg in controls.groupby("ticker")}
        ck = []
        for tk, ev in zip(df["ticker"].to_numpy(), df["event_date"].to_numpy()):
            gg = by_ticker.get(tk)
            if gg is None:
                ck.append(""); continue
            delta = (gg["date"] - pd.Timestamp(ev)).dt.days.abs()
            hit = gg[delta <= control_match_days]
            ck.append(str(hit.iloc[0]["kind"]) if len(hit) else "")
        df["control_kind"] = ck
    df["is_control"] = df["control_kind"] != ""

    # --- censoring ----------------------------------------------------------
    # An event whose outcome window extends past the end of the data has an
    # unobservable label. So does one on a ticker whose listing ended early for
    # a benign reason before the horizon elapsed.
    horizon_end = df["event_date"] + pd.Timedelta(days=horizon_days)
    df["censored"] = (horizon_end > panel_end) & (df["label"] == 0)

    # --- point-in-time eligibility -----------------------------------------
    cap = df["market_cap_proxy"]
    df["eligible"] = (
        (df["close"] >= MIN_PRICE)
        & (cap.isna() | (cap <= MAX_MARKET_CAP))
        & df["has_post_window"]
    )
    return df


def censoring_report(df: pd.DataFrame) -> dict:
    """Numbers that must be stated alongside any performance metric."""
    n = int(len(df))
    pos = int(df["label"].sum())
    cens = int(df["censored"].sum())
    return {
        "n_events": n,
        "n_positive": pos,
        "base_rate": (pos / n) if n else 0.0,
        "n_censored": cens,
        "censored_share": (cens / n) if n else 0.0,
        "n_control": int(df["is_control"].sum()),
        "control_kinds": df.loc[df["is_control"], "control_kind"].value_counts().to_dict(),
        "label_kinds": df.loc[df["label"] == 1, "label_kind"].value_counts().to_dict(),
        "note": ("Negatives include undetected manipulation. Enforcement is a lagged, "
                 "filtered observation of the phenomenon, so precision here is a lower "
                 "bound and recall is measured against detected cases only."),
    }
