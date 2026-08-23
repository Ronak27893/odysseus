"""The screen must not be able to see the future."""
import pandas as pd
import pytest

from ldx.config import FEATURES
from ldx.features import build_features
from ldx.data.synthetic import SyntheticConfig, generate_panel

RETROSPECTIVE = ("retracement", "volume_decay", "offering_within_10d")


def test_screen_columns_exclude_retrospective_features():
    cols = set(FEATURES.model_columns("screen", include_intraday=True))
    for c in RETROSPECTIVE:
        assert c not in cols, f"{c} needs hindsight and must not reach the screen"


def test_forensic_columns_include_retrospective_features():
    cols = set(FEATURES.model_columns("forensic"))
    assert {"retracement", "volume_decay"} <= cols


def test_screen_and_forensic_differ():
    assert (set(FEATURES.model_columns("forensic"))
            - set(FEATURES.model_columns("screen"))) == set(FEATURES.retrospective)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        FEATURES.model_columns("wishful")


def test_offering_prior_is_causal_and_within_is_not():
    """`offering_prior_10d` looks back only; `offering_within_10d` looks ahead."""
    panel = generate_panel(SyntheticConfig(n_tickers=40, seed=3))
    f = build_features(panel, windows=(5,))
    assert "offering_prior_10d" in FEATURES.causal
    assert "offering_within_10d" in FEATURES.retrospective
    # The forward-looking variant must see at least as many offerings.
    assert f["offering_within_10d"].sum() >= f["offering_prior_10d"].sum()


def test_retrospective_features_are_nan_without_post_window():
    """An event at the very end of the data cannot have post-window features."""
    panel = generate_panel(SyntheticConfig(n_tickers=60, seed=11))
    f = build_features(panel, windows=(5,))
    tail = f[~f["has_post_window"]]
    if len(tail):
        assert tail["retracement"].isna().all()
        assert tail["volume_decay"].isna().all()
