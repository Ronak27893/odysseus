"""Unsupervised screen, for sources that cannot supply a positive class.

yfinance is survivor-only: the issuers that were suspended or delisted have no
history to fetch, so there is nothing to train on. That rules out the
supervised model but not the original question, which was whether the
signature is *separable* at all. This module ranks candidate events by how far
they sit from the candidate population on the hypothesised axes.

It is a stated prior, not a fitted model. The weights below were chosen to
follow the hypothesis -- float turnover first, because the claim is that the
move exists to create exit depth -- and were not tuned against any outcome.
Nothing here is validated: with no labels there is no precision to report, and
the confound problem is entirely unaddressed. A widely-covered attention story
will rank exactly like an engineered exit, because on price and volume alone
they are the same object. Read the output as "this name's liquidity profile
changed shape", never as a verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: (feature, weight, direction, log-transform). Direction +1 means larger is
#: more suspicious. Weights are relative and are renormalised over whichever
#: features a given source actually supplies.
DEFAULT_WEIGHTS: tuple[tuple[str, float, int, bool], ...] = (
    ("float_turnover",        3.0, +1, True),   # the hypothesis's operative variable
    ("concentration_ratio",   2.0, +1, True),
    ("peak_relative_volume",  1.5, +1, True),
    ("runup",                 1.5, +1, True),
    ("prior_reverse_splits",  1.0, +1, False),
    ("offering_prior_10d",    1.0, +1, False),  # needs EDGAR; absent under yfinance
    ("news_filing_in_window", 1.5, -1, False),  # needs EDGAR; absent under yfinance
    ("overnight_share",       0.5, +1, False),  # weak alone -- see METHODS.md sect. 6
)

#: Retrospective axes; included only when post-event data exists.
RETROSPECTIVE_WEIGHTS: tuple[tuple[str, float, int, bool], ...] = (
    ("retracement",  2.0, +1, False),
    ("volume_decay", 2.0, -1, True),
)


def robust_z(x: pd.Series, log: bool = False) -> pd.Series:
    """Median/MAD z-score. Robust because these distributions are all tail."""
    v = pd.to_numeric(x, errors="coerce").astype(float)
    if log:
        v = np.log1p(v.clip(lower=0))
    med = v.median()
    mad = (v - med).abs().median()
    scale = mad * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        scale = v.std(ddof=0)
    if not np.isfinite(scale) or scale <= 0:
        return pd.Series(np.zeros(len(v)), index=v.index)
    return ((v - med) / scale).clip(-6, 6)


def score_events(df: pd.DataFrame, include_retrospective: bool = True,
                 weights=DEFAULT_WEIGHTS) -> pd.DataFrame:
    """Return `df` with `signature_score`, per-axis z-scores, and drivers.

    Missing features are skipped and the remaining weights renormalised, so a
    source without EDGAR coverage gets a coherent score over what it does have
    rather than a silent zero for what it lacks.
    """
    if not len(df):
        return df.assign(signature_score=pd.Series(dtype=float))
    out = df.copy()
    active = list(weights)
    if include_retrospective:
        active += list(RETROSPECTIVE_WEIGHTS)

    used, total = [], 0.0
    contrib = pd.DataFrame(index=out.index)
    for name, w, direction, log in active:
        if name not in out.columns:
            continue
        col = pd.to_numeric(out[name], errors="coerce")
        if col.notna().sum() < max(10, int(0.2 * len(out))):
            continue                      # too sparse to normalise meaningfully
        z = robust_z(col, log=log) * direction
        z = z.fillna(0.0)
        contrib[name] = z * w
        used.append(name)
        total += w

    if not used:
        raise ValueError("no usable features for the unsupervised score")
    out["signature_score"] = contrib.sum(axis=1) / total
    for name in used:
        out[f"z_{name}"] = contrib[name] / max(
            next(w for n, w, _, _ in active if n == name), 1e-9)

    # Name the axes actually pushing each event up (positive contribution only).
    cols = list(contrib.columns)
    vals = contrib.to_numpy()
    order = np.argsort(-vals, axis=1)[:, :3]
    drivers = []
    for row_i in range(vals.shape[0]):
        names = [cols[j] for j in order[row_i] if vals[row_i, j] > 0]
        drivers.append(", ".join(names))
    out["drivers"] = drivers
    out.attrs["features_used"] = used
    out.attrs["features_missing"] = [n for n, *_ in active if n not in used]
    return out


def rank(df: pd.DataFrame, top_n: int = 50, asof=None,
         lookback_days: int = 365) -> pd.DataFrame:
    """Score, filter to a recent window, and return the top `top_n`."""
    scored = score_events(df)
    if asof is not None:
        asof = pd.Timestamp(asof)
        scored = scored[(scored["event_date"] <= asof)
                        & (scored["event_date"] >= asof - pd.Timedelta(days=lookback_days))]
    cols = [c for c in ("ticker", "event_date", "signature_score", "drivers",
                        "float_turnover", "concentration_ratio", "runup",
                        "retracement", "volume_decay", "overnight_share",
                        "peak_relative_volume", "prior_reverse_splits",
                        "float_shares", "close", "suspected_unadjusted_split")
            if c in scored.columns]
    out = scored.sort_values("signature_score", ascending=False).head(top_n)
    out = out[cols].reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
