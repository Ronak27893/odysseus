"""Candidate-event detection and feature computation.

One row per (ticker, event_date, window). The primary window is 5 trading
days; 3d and 10d are computed at the same anchor so the robustness check
compares measurement scales on the *same* events rather than on three
different event sets.

Causality note (important, and not spelled out by the feature list itself):
features 4 (`retracement`) and 5 (`volume_decay`) need 20 trading days after
the window, and an offering filed after the ramp is not knowable during it.
Those columns are marked retrospective and excluded from the screen model, so
a real-time ranking cannot silently use hindsight. See config.FeatureSpec.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CANDIDATE_MIN_CONCENTRATION_RATIO, EVENT_REFRACTORY_DAYS, EVENT_WINDOWS,
    MIN_TRAILING_DAYS, POST_WINDOW_DAYS, PRE_BASELINE_DAYS, PRE_VOLUME_DAYS,
    PRIMARY_WINDOW, TRAILING_LIQUIDITY_DAYS,
)
#: Minimum absolute log run-up over the leg before overnight_share is defined.
MIN_LEG_RETURN = 0.05
#: Minimum run-up, as a fraction of the pre-window price level, before
#: `retracement` is defined. Below this the denominator is noise.
MIN_RUNUP_ABS = 0.10


def _rolling_median(x: np.ndarray, k: int) -> np.ndarray:
    """m[e] = median(x[e-k+1 : e+1]); NaN where the window is incomplete."""
    n = len(x)
    out = np.full(n, np.nan)
    if n >= k:
        win = np.lib.stride_tricks.sliding_window_view(x, k)
        out[k - 1:] = np.median(win, axis=1)
    return out


def _trailing_sum(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing sum over min(k, available) plus the number of days used."""
    n = len(x)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - k + 1)
    total = csum[idx + 1] - csum[lo]
    used = idx - lo + 1
    return total, used


def concentration_ratio(win_dv: np.ndarray, trail_dv: np.ndarray,
                        trail_days: np.ndarray, window: int) -> np.ndarray:
    """Window liquidity share divided by its proportional share.

    1.0 under flat volume for ANY history length, so unlike the raw ratio it
    does not manufacture candidates for newly-listed tickers.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = window / np.maximum(trail_days, 1)
        raw = np.where(trail_dv > 0, win_dv / trail_dv, np.nan)
        return raw / expected


def detect_candidates(close: np.ndarray, volume: np.ndarray,
                      window: int = PRIMARY_WINDOW,
                      min_ratio: float = CANDIDATE_MIN_CONCENTRATION_RATIO,
                      refractory: int = EVENT_REFRACTORY_DAYS) -> list[int]:
    """Return anchor indices (window END) of candidate events for one ticker.

    Deliberately high-recall: the model does the ranking, this stage only
    avoids scoring all ~1400 days of every ticker. Non-maximum suppression
    collapses one episode into a single anchor.
    """
    n = len(close)
    if n < window + PRE_BASELINE_DAYS + 2:
        return []
    dv = close * volume
    win_dv, _ = _trailing_sum(dv, window)
    trail_dv, trail_days = _trailing_sum(dv, TRAILING_LIQUIDITY_DAYS)
    ratio = concentration_ratio(win_dv, trail_dv, trail_days, window)
    eligible = (trail_days >= MIN_TRAILING_DAYS) & np.isfinite(ratio) & (ratio >= min_ratio)
    idx = np.flatnonzero(eligible)
    if idx.size == 0:
        return []
    accepted: list[int] = []
    for i in idx[np.argsort(-ratio[idx])]:         # strongest first
        if all(abs(int(i) - a) >= refractory for a in accepted):
            accepted.append(int(i))
    return sorted(accepted)


def _ticker_features(ticker: str, g: pd.DataFrame, panel,
                     windows: tuple[int, ...],
                     split_flags: set,
                     ipo_date, rev_split_dates: np.ndarray,
                     offering_dates: np.ndarray,
                     intraday: pd.DataFrame | None,
                     news_dates: np.ndarray) -> list[dict]:
    g = g.sort_values("date")
    dates = g["date"].to_numpy()
    close = g["close"].to_numpy(dtype=float)
    open_ = g["open"].to_numpy(dtype=float)
    volume = g["volume"].to_numpy(dtype=float)
    n = len(close)
    dv = close * volume

    anchors = detect_candidates(close, volume)
    if not anchors:
        return []

    trail_dv, trail_days = _trailing_sum(dv, TRAILING_LIQUIDITY_DAYS)
    med_close_pre = _rolling_median(close, PRE_BASELINE_DAYS)
    med_vol_pre = _rolling_median(volume, PRE_VOLUME_DAYS)
    med_vol_20 = _rolling_median(volume, PRE_BASELINE_DAYS)
    # Overnight / intraday decomposition of each day's log return.
    gap = np.full(n, np.nan)
    gap[1:] = np.log(np.maximum(open_[1:], 1e-9) / np.maximum(close[:-1], 1e-9))
    total = np.full(n, np.nan)
    total[1:] = np.log(np.maximum(close[1:], 1e-9) / np.maximum(close[:-1], 1e-9))

    if intraday is not None and len(intraday):
        intr = intraday.set_index("date")
        didx = pd.DatetimeIndex(dates)
        auction = intr["closing_auction_share"].reindex(didx).to_numpy(dtype=float)
        small = intr["small_trade_share"].reindex(didx).to_numpy(dtype=float)
    else:
        auction = np.full(n, np.nan)
        small = np.full(n, np.nan)

    rows: list[dict] = []
    for t in anchors:
        anchor_date = pd.Timestamp(dates[t])
        for W in windows:
            ws = t - W + 1
            if ws < 1:
                continue
            pre_end = ws - 1                       # last pre-window day
            if pre_end < 0:
                continue
            post_end = t + POST_WINDOW_DAYS
            has_post = post_end < n

            w_vol = float(volume[ws:t + 1].sum())
            w_dv = float(dv[ws:t + 1].sum())
            conc = w_dv / trail_dv[t] if trail_dv[t] > 0 else np.nan
            conc_ratio = (conc / (W / max(int(trail_days[t]), 1))
                          if np.isfinite(conc) else np.nan)

            base_close = med_close_pre[pre_end]
            peak_rel = int(np.argmax(close[ws:t + 1]))
            peak_idx = ws + peak_rel
            peak = float(close[peak_idx])
            runup = peak / base_close if base_close and base_close > 0 else np.nan

            # --- retrospective (needs 20 trading days after the window) ----
            if has_post:
                post_close = float(np.median(close[t + 1:post_end + 1]))
                post_vol = float(np.median(volume[t + 1:post_end + 1]))
                # Require a run-up that is real relative to the price level;
                # otherwise (peak - base) is noise and the ratio explodes.
                denom = peak - base_close
                retracement = ((peak - post_close) / denom
                               if np.isfinite(denom) and denom > MIN_RUNUP_ABS * base_close
                               else np.nan)
                pre_vol_base = med_vol_pre[pre_end]
                volume_decay = (post_vol / pre_vol_base
                                if np.isfinite(pre_vol_base) and pre_vol_base > 0 else np.nan)
            else:
                retracement = np.nan
                volume_decay = np.nan

            # --- overnight share over the run-up leg -----------------------
            leg = slice(ws, peak_idx + 1)
            leg_total = float(np.nansum(total[leg]))
            leg_gap = float(np.nansum(gap[leg]))
            overnight_share = (leg_gap / leg_total
                               if abs(leg_total) >= MIN_LEG_RETURN else np.nan)

            # --- float, point-in-time at window start ----------------------
            float_shares = panel.float_as_of(ticker, pd.Timestamp(dates[ws]))
            float_turnover = (w_vol / float_shares
                              if float_shares and float_shares > 0 else np.nan)

            pre_vol_20 = med_vol_20[pre_end]
            peak_rel_vol = (float(volume[ws:t + 1].max()) / pre_vol_20
                            if np.isfinite(pre_vol_20) and pre_vol_20 > 0 else np.nan)

            days_since_ipo = ((anchor_date - ipo_date).days
                              if ipo_date is not None and pd.notna(ipo_date) else np.nan)
            n_prior_rev = int((rev_split_dates < np.datetime64(anchor_date)).sum())

            a10 = np.datetime64(anchor_date + pd.Timedelta(days=10))
            a_m10 = np.datetime64(anchor_date - pd.Timedelta(days=10))
            ws_date = np.datetime64(pd.Timestamp(dates[ws]))
            anchor64 = np.datetime64(anchor_date)
            offering_within = int(((offering_dates >= ws_date) & (offering_dates <= a10)).any())
            offering_prior = int(((offering_dates >= a_m10) & (offering_dates <= anchor64)).any())
            # Was there filed news during the window? `overnight_share` is not
            # single-signed on its own: scheduled corporate news is released
            # outside market hours, so genuine-news events gap HARDER than a
            # ramp, while ordinary noise and attention-driven moves gap less.
            # Gapping WITHOUT a filing is the discriminating cell.
            has_news = int(((news_dates >= ws_date) & (news_dates <= anchor64)).any())

            w_vol_arr = volume[ws:t + 1]
            wsum = w_vol_arr.sum()
            auc = (float(np.nansum(auction[ws:t + 1] * w_vol_arr) / wsum)
                   if wsum > 0 and np.isfinite(auction[ws:t + 1]).any() else np.nan)
            sml = (float(np.nansum(small[ws:t + 1] * w_vol_arr) / wsum)
                   if wsum > 0 and np.isfinite(small[ws:t + 1]).any() else np.nan)

            rows.append({
                "ticker": ticker,
                "event_date": anchor_date,
                "window": W,
                "window_start": pd.Timestamp(dates[ws]),
                # --- causal features ---------------------------------------
                "liquidity_concentration": conc,
                "concentration_ratio": conc_ratio,
                "float_turnover": float_turnover,
                "runup": runup,
                "overnight_share": (float(np.clip(overnight_share, -1.0, 2.0))
                                    if np.isfinite(overnight_share) else np.nan),
                "days_since_ipo": days_since_ipo,
                "days_since_ipo_log": (float(np.log1p(days_since_ipo))
                                       if np.isfinite(days_since_ipo) else np.nan),
                "offering_prior_10d": offering_prior,
                "news_filing_in_window": has_news,
                "gap_without_news": (float(np.clip(overnight_share, -1.0, 2.0)) * (1 - has_news)
                                     if np.isfinite(overnight_share) else np.nan),
                "prior_reverse_splits": n_prior_rev,
                "closing_auction_share": auc,
                "small_trade_share": sml,
                # --- retrospective ------------------------------------------
                "retracement": (float(np.clip(retracement, -2.0, 3.0))
                                if np.isfinite(retracement) else np.nan),
                "volume_decay": volume_decay,
                "offering_within_10d": offering_within,
                "has_post_window": bool(has_post),
                # --- diagnostics --------------------------------------------
                "peak_relative_volume": peak_rel_vol,
                "window_dollar_volume": w_dv,
                "float_shares": float_shares if float_shares else np.nan,
                "market_cap_proxy": (close[t] * float_shares) if float_shares else np.nan,
                "close": float(close[t]),
                "trailing_days": int(trail_days[t]),
                "suspected_unadjusted_split": bool(
                    any(abs((pd.Timestamp(d) - anchor_date).days) <= 30 for d in split_flags)),
            })
    return rows


def build_features(panel, windows: tuple[int, ...] = EVENT_WINDOWS,
                   split_flags: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute the feature store for every candidate event in the panel."""
    sec = panel.securities.set_index("ticker")
    rev = panel.corporate_actions
    rev = rev[rev["kind"] == "reverse_split"] if len(rev) else rev
    fil = panel.filings
    offerings = fil[fil["form"].isin(["424B5", "S-3", "S-1"])] if len(fil) else fil
    news = fil[fil["form"].isin(["8-K"])] if len(fil) else fil

    flags_by_ticker: dict[str, set] = {}
    if split_flags is not None and len(split_flags):
        for tk, gg in split_flags.groupby("ticker"):
            flags_by_ticker[tk] = set(pd.to_datetime(gg["date"]))

    intraday_by_ticker = {}
    if panel.intraday is not None and len(panel.intraday):
        for tk, gg in panel.intraday.groupby("ticker", sort=False):
            intraday_by_ticker[tk] = gg

    rev_by_ticker = ({tk: gg["date"].to_numpy() for tk, gg in rev.groupby("ticker")}
                     if len(rev) else {})
    off_by_ticker = ({tk: gg["date"].to_numpy() for tk, gg in offerings.groupby("ticker")}
                     if len(offerings) else {})
    news_by_ticker = ({tk: gg["date"].to_numpy() for tk, gg in news.groupby("ticker")}
                      if len(news) else {})

    empty_dt = np.array([], dtype="datetime64[ns]")
    rows: list[dict] = []
    for ticker, g in panel.bars.groupby("ticker", sort=False):
        ipo = sec["ipo_date"].get(ticker) if "ipo_date" in sec.columns else None
        rows.extend(_ticker_features(
            ticker, g, panel, windows,
            flags_by_ticker.get(ticker, set()),
            ipo,
            rev_by_ticker.get(ticker, empty_dt),
            off_by_ticker.get(ticker, empty_dt),
            intraday_by_ticker.get(ticker),
            news_by_ticker.get(ticker, empty_dt),
        ))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["event_date", "ticker", "window"])
    return out.reset_index(drop=True)
