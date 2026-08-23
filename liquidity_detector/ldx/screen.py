"""Ranked daily screen.

Uses the SCREEN model (causal features only), because a ranking that needs 20
days of hindsight is not a screen. Every row carries the features that drove
its score, so the output reads as a signature to check rather than a verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

_DISPLAY = ["ticker", "event_date", "score", "rank", "drivers",
            "liquidity_concentration", "float_turnover", "runup",
            "overnight_share", "offering_prior_10d", "prior_reverse_splits",
            "peak_relative_volume", "float_shares", "close",
            "suspected_unadjusted_split", "control_kind", "label"]


def _logistic_drivers(pipe: Pipeline, X: pd.DataFrame, columns: list[str],
                      top: int = 3) -> list[str]:
    """Per-event contribution = standardised value x coefficient."""
    try:
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
    except (AttributeError, KeyError):
        return [""] * len(X)
    Z = pre.transform(X[columns])
    Z = np.asarray(Z, dtype=float)
    names: list[str] = []
    for name, _, cols in pre.transformers_:
        if name != "remainder":
            names.extend(cols)
    contrib = Z * clf.coef_[0]
    out = []
    for row in contrib:
        order = np.argsort(-row)[:top]
        out.append(", ".join(f"{names[i]}{'+' if row[i] >= 0 else '-'}"
                             for i in order if np.isfinite(row[i]) and row[i] > 0))
    return out


def build_screen(df: pd.DataFrame, pipe: Pipeline, columns: list[str],
                 asof: pd.Timestamp | None = None, lookback_days: int = 365,
                 top_n: int = 50) -> pd.DataFrame:
    """Score recent events and return the top `top_n`, most suspicious first."""
    if not len(df):
        return pd.DataFrame(columns=_DISPLAY)
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp(df["event_date"].max())
    recent = df[(df["event_date"] <= asof)
                & (df["event_date"] >= asof - pd.Timedelta(days=lookback_days))].copy()
    if not len(recent):
        return pd.DataFrame(columns=_DISPLAY)
    recent["score"] = pipe.predict_proba(recent[columns])[:, 1]
    recent = recent.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    recent["rank"] = np.arange(1, len(recent) + 1)
    recent["drivers"] = _logistic_drivers(pipe, recent, columns)
    cols = [c for c in _DISPLAY if c in recent.columns]
    return recent[cols]
