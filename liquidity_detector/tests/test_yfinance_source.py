"""yfinance adapter, exercised against a stub so the tests need no network."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from ldx.data.integrity import audit_panel
from ldx.data.yfinance_source import YFinanceSource, load_universe_file

DATES = pd.bdate_range("2020-01-01", periods=300)


def _frame(close_scale=1.0, adj_offset=0.0):
    n = len(DATES)
    close = np.linspace(10, 20, n) * close_scale
    return pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.02, "Low": close * 0.98,
        "Close": close,
        # Deliberately different, so a test can prove which column is read.
        "Adj Close": close - adj_offset,
        "Volume": np.full(n, 100_000.0),
    }, index=DATES)


def _install_stub(monkeypatch, tickers=("AAA", "BBB"), splits=None, shares=None,
                  multiindex=True, missing=()):
    frames = {t: _frame(adj_offset=5.0) for t in tickers if t not in missing}
    if multiindex and frames:
        raw = pd.concat(frames, axis=1)
    elif frames:
        raw = next(iter(frames.values()))
    else:
        raw = pd.DataFrame()

    class _Ticker:
        def __init__(self, name):
            self.name = name

        @property
        def splits(self):
            return splits if splits is not None else pd.Series(dtype=float)

        def get_shares_full(self, start=None, end=None):
            return shares if shares is not None else pd.Series(dtype=float)

    stub = types.ModuleType("yfinance")
    stub.download = lambda *a, **k: raw
    stub.Ticker = _Ticker
    monkeypatch.setitem(sys.modules, "yfinance", stub)
    return stub


def _source(tmp_path, tickers=("AAA", "BBB")):
    return YFinanceSource(list(tickers), start="2020-01-01", end="2021-03-01",
                          cache_dir=tmp_path / "cache", use_shares_history=True)


def test_reads_close_not_adj_close(monkeypatch, tmp_path):
    """Dividend-adjusted prices would distort price level and dollar volume."""
    _install_stub(monkeypatch)
    bars = _source(tmp_path).load_bars()
    aaa = bars[bars.ticker == "AAA"].sort_values("date")
    expected = _frame(adj_offset=5.0)["Close"].to_numpy()
    np.testing.assert_allclose(aaa["close"].to_numpy(), expected)


def test_handles_flat_single_ticker_frame(monkeypatch, tmp_path):
    _install_stub(monkeypatch, tickers=("AAA",), multiindex=False)
    bars = YFinanceSource(["AAA"], start="2020-01-01", end="2021-03-01",
                          cache_dir=tmp_path / "c").load_bars()
    assert set(bars["ticker"]) == {"AAA"}
    assert len(bars) == len(DATES)


def test_missing_tickers_are_recorded_not_dropped_silently(monkeypatch, tmp_path):
    _install_stub(monkeypatch, tickers=("AAA", "BBB"), missing=("BBB",))
    src = _source(tmp_path)
    src.load_bars()
    assert "BBB" in src.missing


def test_all_missing_raises_with_survivorship_explanation(monkeypatch, tmp_path):
    _install_stub(monkeypatch, tickers=("AAA",), missing=("AAA",))
    src = YFinanceSource(["AAA"], cache_dir=tmp_path / "c")
    with pytest.raises(RuntimeError, match="survivorship"):
        src.load_bars()


def test_reverse_split_ratio_is_normalised(monkeypatch, tmp_path):
    """yfinance reports a 1-for-10 reverse split as 0.1."""
    splits = pd.Series([0.1], index=[pd.Timestamp("2020-06-01")])
    _install_stub(monkeypatch, splits=splits)
    actions, _ = _source(tmp_path).load_actions_and_shares(["AAA"])
    assert len(actions) == 1
    assert actions.iloc[0]["kind"] == "reverse_split"
    assert actions.iloc[0]["ratio"] == pytest.approx(10.0)


def test_forward_split_kept_as_is(monkeypatch, tmp_path):
    splits = pd.Series([4.0], index=[pd.Timestamp("2020-06-01")])
    _install_stub(monkeypatch, splits=splits)
    actions, _ = _source(tmp_path).load_actions_and_shares(["AAA"])
    assert actions.iloc[0]["kind"] == "split"
    assert actions.iloc[0]["ratio"] == pytest.approx(4.0)


def test_share_history_is_a_time_series_not_a_snapshot(monkeypatch, tmp_path):
    """The point of get_shares_full: a denominator that changes over time."""
    shares = pd.Series([1e7, 2e7],
                       index=[pd.Timestamp("2020-01-02"), pd.Timestamp("2020-09-01")])
    _install_stub(monkeypatch, shares=shares)
    _, sh = _source(tmp_path).load_actions_and_shares(["AAA"])
    assert len(sh) == 2
    rep_dates = sorted(sh["date"].tolist())
    assert rep_dates[0] < rep_dates[1]


def test_panel_declares_survivorship_and_share_basis(monkeypatch, tmp_path):
    _install_stub(monkeypatch)
    panel = _source(tmp_path).load()
    assert "survivor-only" in panel.survivorship_note
    # Shares outstanding is not float; the adapter must not conflate them.
    assert panel.share_count_basis == "shares_outstanding"


def test_audit_names_the_consequence_for_a_survivor_only_panel(monkeypatch, tmp_path):
    _install_stub(monkeypatch)
    panel = _source(tmp_path).load()
    rep = audit_panel(panel, strict=False)
    assert not rep.ok
    joined = " ".join(rep.fatal)
    assert "unsupervised screen remains valid" in joined
    assert "must not" in joined


def test_universe_file_txt_and_csv(tmp_path):
    txt = tmp_path / "u.txt"
    txt.write_text("aaa\n# comment\n\nbbb\n")
    assert load_universe_file(txt) == ["AAA", "BBB"]
    csv = tmp_path / "u.csv"
    csv.write_text("ticker,name\nccc,C Corp\nddd,D Corp\n")
    assert load_universe_file(csv) == ["CCC", "DDD"]
