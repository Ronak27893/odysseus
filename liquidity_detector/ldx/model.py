"""Models. Logistic regression first, so coefficients mean something.

The spec's ordering is deliberate and is followed here: an interpretable
linear model is fitted and reported before any boosted ensemble, so that a
reader can see WHICH part of the hypothesised signature is doing the work
rather than only that something is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

#: Strictly-positive, heavy-tailed features. Logged before standardising so a
#: single 124x float-turnover print does not dominate the linear fit.
HEAVY_TAILED = ("liquidity_concentration", "concentration_ratio", "float_turnover",
                "runup", "volume_decay", "peak_relative_volume")


def _log1p_safe(x):
    return np.log1p(np.clip(x, 0.0, None))


def build_logistic(columns: list[str], C: float = 1.0) -> Pipeline:
    """Interpretable baseline: log-transform, impute, standardise, logit."""
    heavy = [c for c in columns if c in HEAVY_TAILED]
    plain = [c for c in columns if c not in HEAVY_TAILED]
    heavy_pipe = Pipeline([
        ("log", FunctionTransformer(_log1p_safe, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    plain_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer([("heavy", heavy_pipe, heavy), ("plain", plain_pipe, plain)],
                            remainder="drop")
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                   solver="lbfgs")),
    ])


def build_gbm(columns: list[str], seed: int = 0) -> Pipeline:
    """Gradient boosting. Handles NaN natively; no scaling needed."""
    return Pipeline([
        ("select", FunctionTransformer(lambda X: X[columns], feature_names_out="one-to-one")),
        ("clf", HistGradientBoostingClassifier(
            max_depth=3, max_iter=250, learning_rate=0.06,
            min_samples_leaf=15, l2_regularization=1.0,
            class_weight="balanced", random_state=seed)),
    ])


MODELS = {"logistic": build_logistic, "gbm": build_gbm}


@dataclass
class Coefficients:
    """Standardised logistic coefficients, in spec order."""
    names: list[str]
    values: list[float]
    odds_ratios: list[float]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.names,
            "coef_std": self.values,
            "odds_ratio_per_sd": self.odds_ratios,
        }).sort_values("coef_std", key=abs, ascending=False).reset_index(drop=True)


def logistic_coefficients(pipe: Pipeline, columns: list[str]) -> Coefficients:
    """Recover coefficients aligned to the original column names."""
    pre: ColumnTransformer = pipe.named_steps["pre"]
    order: list[str] = []
    for name, _, cols in pre.transformers_:
        if name != "remainder":
            order.extend(cols)
    coefs = pipe.named_steps["clf"].coef_[0]
    return Coefficients(
        names=order,
        values=[float(c) for c in coefs],
        odds_ratios=[float(np.exp(c)) for c in coefs],
    )
