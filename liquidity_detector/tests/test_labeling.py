"""Label horizon, censoring, and control tagging."""
import pandas as pd
import pytest

from ldx.config import LABEL_HORIZON_DAYS
from ldx.data.base import Panel
from ldx.labeling import attach_labels, censoring_report


def _panel(label_offset_days=None, control=False, panel_end="2022-12-31"):
    dates = pd.bdate_range("2020-01-01", panel_end)
    bars = pd.DataFrame({"ticker": "AAA", "date": dates, "open": 10.0, "high": 10.0,
                         "low": 10.0, "close": 10.0, "volume": 1000.0})
    sec = pd.DataFrame([{"ticker": "AAA", "first_date": dates[0], "last_date": dates[-1],
                         "ipo_date": dates[0]}])
    labels = pd.DataFrame(columns=["ticker", "date", "kind"])
    if label_offset_days is not None:
        labels = pd.DataFrame([{"ticker": "AAA",
                                "date": pd.Timestamp("2021-01-04") + pd.Timedelta(days=label_offset_days),
                                "kind": "sec_suspension"}])
    controls = pd.DataFrame(columns=["ticker", "date", "kind"])
    if control:
        controls = pd.DataFrame([{"ticker": "AAA", "date": pd.Timestamp("2021-01-04"),
                                  "kind": "short_squeeze"}])
    return Panel(bars=bars, securities=sec, label_events=labels, control_events=controls)


def _features(event_date="2021-01-04"):
    return pd.DataFrame([{
        "ticker": "AAA", "event_date": pd.Timestamp(event_date), "window": 5,
        "market_cap_proxy": 1e8, "close": 10.0, "has_post_window": True,
    }])


def test_label_inside_horizon_is_positive():
    out = attach_labels(_features(), _panel(label_offset_days=LABEL_HORIZON_DAYS - 1))
    assert out["label"].iloc[0] == 1
    assert out["label_kind"].iloc[0] == "sec_suspension"


def test_label_outside_horizon_is_negative():
    out = attach_labels(_features(), _panel(label_offset_days=LABEL_HORIZON_DAYS + 30))
    assert out["label"].iloc[0] == 0


def test_label_before_event_does_not_count():
    """An outcome preceding the event cannot have been predicted by it."""
    out = attach_labels(_features(), _panel(label_offset_days=-30))
    assert out["label"].iloc[0] == 0


def test_event_near_panel_end_is_censored_not_negative():
    """Unobservable outcomes must not be taught to the model as safe."""
    panel = _panel(panel_end="2021-03-31")
    out = attach_labels(_features("2021-01-04"), panel)
    assert bool(out["censored"].iloc[0]) is True


def test_observed_negative_is_not_censored():
    out = attach_labels(_features("2021-01-04"), _panel(panel_end="2022-12-31"))
    assert bool(out["censored"].iloc[0]) is False


def test_control_is_tagged():
    out = attach_labels(_features(), _panel(control=True))
    assert bool(out["is_control"].iloc[0]) is True
    assert out["control_kind"].iloc[0] == "short_squeeze"


def test_censoring_report_states_the_caveat():
    out = attach_labels(_features(), _panel(label_offset_days=10))
    rep = censoring_report(out)
    assert rep["n_positive"] == 1
    assert "undetected manipulation" in rep["note"].lower()
