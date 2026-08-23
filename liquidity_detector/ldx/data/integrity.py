"""Data-integrity audits that must pass before any feature is computed.

Two failure modes named in the spec are handled here because both silently
manufacture the exact signature we are hunting:

1. Survivorship bias. Suspended issuers get delisted and vendor history
   disappears, so a universe drawn from *currently listed* names has the
   positive class removed from it by construction.
2. Unadjusted prices. A 1-for-10 reverse split is a 10x "run-up" followed by
   a volume collapse -- indistinguishable from a ramp on unadjusted data.

Neither is checked by asserting a vendor flag. Both are measured from the
bars themselves.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from ..config import (MAX_SUSPECT_SPLIT_TICKER_SHARE, MIN_DEAD_TICKER_SHARE,
                      SPLIT_SUSPECT_RATIO)

#: Ratios a real split is plausibly declared at. An unadjusted split lands on
#: one of these; a genuine market move generally does not.
COMMON_SPLIT_RATIOS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200)
_RATIO_TOLERANCE = 0.06        # 6% around the nominal ratio
_PERSISTENCE_DAYS = 3          # a split is a level shift, not a spike
_REVERSION_TOLERANCE = 0.35    # if it retraces this much of the jump it is a move, not a split
#: A split leaves dollar volume continuous: share volume must scale by the
#: inverse of the price factor. Tolerance is generous (a factor of 2.5 either
#: way) because it only needs to separate "volume moved inversely" from
#: "volume exploded alongside price", which is what a ramp does.
_VOLUME_CONSISTENCY_TOLERANCE = 2.5


class IntegrityError(RuntimeError):
    """Raised when a panel is unfit for modelling."""


@dataclass
class IntegrityReport:
    source_name: str
    n_tickers: int = 0
    n_rows: int = 0
    date_min: str = ""
    date_max: str = ""
    dead_ticker_share: float = 0.0
    n_dead_tickers: int = 0
    suspected_unadjusted_splits: int = 0
    tickers_with_suspected_splits: list[str] = field(default_factory=list)
    float_history_is_snapshot: bool = False
    float_coverage: float = 0.0
    n_nonpositive_prices: int = 0
    n_duplicate_rows: int = 0
    n_zero_volume_days: int = 0
    fatal: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.fatal

    def to_dict(self) -> dict:
        return asdict(self) | {"ok": self.ok}

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def summary(self) -> str:
        lines = [
            f"integrity: source={self.source_name} tickers={self.n_tickers} rows={self.n_rows}",
            f"  window            {self.date_min} .. {self.date_max}",
            f"  dead tickers      {self.n_dead_tickers} ({self.dead_ticker_share:.1%})",
            f"  suspected splits  {self.suspected_unadjusted_splits}",
            f"  float coverage    {self.float_coverage:.1%}"
            + ("  [SNAPSHOT, not point-in-time]" if self.float_history_is_snapshot else ""),
        ]
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        for f in self.fatal:
            lines.append(f"  FATAL {f}")
        return "\n".join(lines)


def detect_unadjusted_splits(bars: pd.DataFrame,
                             corporate_actions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return rows flagged as suspected *unadjusted* split events.

    A day is flagged when the close-to-close ratio sits near a common split
    ratio, persists (does not retrace) over the following days, and moves
    volume in the opposite direction by a compatible factor -- and no declared
    corporate action explains it. Declared actions are matched within +/-3
    calendar days, so a properly-adjusted series with known splits stays
    clean.
    """
    out: list[dict] = []
    known: set[tuple[str, pd.Timestamp]] = set()
    if corporate_actions is not None and len(corporate_actions):
        for _, r in corporate_actions.iterrows():
            for offset in range(-3, 4):
                known.add((r["ticker"], pd.Timestamp(r["date"]) + pd.Timedelta(days=offset)))

    for ticker, g in bars.groupby("ticker", sort=False):
        g = g.sort_values("date")
        close = g["close"].to_numpy(dtype=float)
        vol = g["volume"].to_numpy(dtype=float)
        dates = g["date"].to_numpy()
        if len(close) < _PERSISTENCE_DAYS + 6:
            continue
        ratio = np.divide(close[:-1], close[1:],
                          out=np.full(len(close) - 1, np.nan),
                          where=close[1:] > 0)
        for i, r in enumerate(ratio):
            if not np.isfinite(r) or r <= 0:
                continue
            # Forward split -> r ~ k (price falls). Reverse split -> r ~ 1/k.
            factor, is_reverse = (r, False) if r >= 1 else (1.0 / r, True)
            if factor < SPLIT_SUSPECT_RATIO:
                continue
            match = next((k for k in COMMON_SPLIT_RATIOS
                          if abs(factor - k) / k <= _RATIO_TOLERANCE), None)
            if match is None:
                continue
            day = pd.Timestamp(dates[i + 1])
            if (ticker, day) in known:
                continue  # declared corporate action explains it
            # Persistence: a split is a permanent level shift.
            tail = close[i + 1: i + 1 + _PERSISTENCE_DAYS + 1]
            if len(tail) < 2:
                continue
            pre, post = close[i], float(np.median(tail))
            jump = np.log(post / pre) if pre > 0 and post > 0 else np.nan
            expected = -np.log(match) if not is_reverse else np.log(match)
            if not np.isfinite(jump) or expected == 0:
                continue
            if abs(jump - expected) / abs(expected) > _REVERSION_TOLERANCE:
                continue  # retraced -> a price move, not a split
            # Volume must scale inversely with the price factor. This is the
            # test a ramp fails: during a ramp price rises AND volume rises,
            # whereas a 1-for-k reverse split raises price and cuts volume by
            # ~k, leaving dollar volume continuous.
            v_pre = float(np.median(vol[max(0, i - 10): i + 1]))
            v_post = float(np.median(vol[i + 1: i + 11]))
            vol_ratio = (v_post / v_pre) if v_pre > 0 else np.nan
            if not np.isfinite(vol_ratio) or vol_ratio <= 0:
                continue
            expected_vol_ratio = (1.0 / match) if is_reverse else float(match)
            if abs(np.log(vol_ratio / expected_vol_ratio)) > np.log(_VOLUME_CONSISTENCY_TOLERANCE):
                continue  # volume did not move inversely -> price move, not a split
            out.append({
                "ticker": ticker, "date": day, "price_factor": float(factor),
                "matched_ratio": int(match), "is_reverse": bool(is_reverse),
                "volume_ratio": float(vol_ratio) if np.isfinite(vol_ratio) else np.nan,
            })
    return pd.DataFrame(out, columns=["ticker", "date", "price_factor",
                                      "matched_ratio", "is_reverse", "volume_ratio"])


def audit_panel(panel, strict: bool = True) -> IntegrityReport:
    """Audit a Panel. Raises IntegrityError on fatal findings when strict."""
    bars = panel.bars
    rep = IntegrityReport(source_name=getattr(panel, "source_name", "unknown"))
    rep.n_tickers = bars["ticker"].nunique()
    rep.n_rows = int(len(bars))
    if rep.n_rows:
        rep.date_min = str(pd.Timestamp(bars["date"].min()).date())
        rep.date_max = str(pd.Timestamp(bars["date"].max()).date())

    # --- survivorship -----------------------------------------------------
    dead = panel.dead_tickers()
    rep.n_dead_tickers = len(dead)
    rep.dead_ticker_share = (len(dead) / rep.n_tickers) if rep.n_tickers else 0.0
    if rep.dead_ticker_share < MIN_DEAD_TICKER_SHARE:
        msg = (f"universe contains {rep.dead_ticker_share:.1%} terminated securities "
               f"(< {MIN_DEAD_TICKER_SHARE:.0%}); a survivor-only panel removes the "
               f"positive class by construction")
        note = getattr(panel, "survivorship_note", None)
        if note:
            # The source already declared itself survivor-only. Say what that
            # rules out, precisely, instead of repeating a generic failure.
            msg += (f". Source declares: {note} Supervised metrics from this panel "
                    f"would be measured against a negative class only and must not "
                    f"be reported; the unsupervised screen remains valid")
        rep.fatal.append(msg)

    # --- split adjustment -------------------------------------------------
    flagged = detect_unadjusted_splits(bars, panel.corporate_actions)
    rep.suspected_unadjusted_splits = int(len(flagged))
    rep.tickers_with_suspected_splits = sorted(flagged["ticker"].unique().tolist())[:50]
    if len(flagged):
        share = flagged["ticker"].nunique() / max(rep.n_tickers, 1)
        msg = (f"{len(flagged)} suspected UNADJUSTED split(s) across "
               f"{flagged['ticker'].nunique()} ticker(s) ({share:.1%} of the universe); "
               f"reverse splits reproduce the run-up/volume-collapse signature exactly")
        if share > MAX_SUSPECT_SPLIT_TICKER_SHARE:
            rep.fatal.append(msg + " -- feed appears systematically unadjusted")
        else:
            rep.warnings.append(msg + " -- affected events are flagged, not dropped")

    # --- float point-in-timeness -----------------------------------------
    fh = panel.float_history
    if len(fh):
        per_ticker = fh.groupby("ticker")["date"].nunique()
        rep.float_coverage = float(per_ticker.reindex(panel.tickers).notna().mean())
        rep.float_history_is_snapshot = bool((per_ticker <= 1).all())
        if rep.float_history_is_snapshot:
            rep.warnings.append(
                "float_history has a single record per ticker: this is a snapshot, not "
                "point-in-time. float_turnover will be biased wherever share count changed")
    else:
        rep.float_coverage = 0.0
        rep.warnings.append("no float history supplied; float_turnover will be NaN")

    # --- basic hygiene ----------------------------------------------------
    rep.n_nonpositive_prices = int((bars["close"] <= 0).sum())
    rep.n_duplicate_rows = int(bars.duplicated(subset=["ticker", "date"]).sum())
    rep.n_zero_volume_days = int((bars["volume"] <= 0).sum())
    if rep.n_nonpositive_prices:
        rep.fatal.append(f"{rep.n_nonpositive_prices} non-positive close price(s)")
    if rep.n_duplicate_rows:
        rep.fatal.append(f"{rep.n_duplicate_rows} duplicate (ticker, date) row(s)")

    if strict and rep.fatal:
        raise IntegrityError(rep.summary())
    return rep
