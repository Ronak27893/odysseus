"""Central configuration: windows, thresholds, paths.

Nothing here is tuned on out-of-sample data. Values are stated so a reviewer
can see exactly what was chosen a priori versus fitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Trading-day conventions.
TRADING_DAYS_YEAR = 252
TRAILING_LIQUIDITY_DAYS = 2 * TRADING_DAYS_YEAR  # "trailing 2y dollar volume"

#: Event windows to compute every feature over. The spec names 5d as primary
#: and asks for 3d and 10d as robustness checks.
EVENT_WINDOWS = (3, 5, 10)
PRIMARY_WINDOW = 5

#: Baseline / follow-up horizons used by the features.
PRE_BASELINE_DAYS = 20    # pre-window median close and volume
PRE_VOLUME_DAYS = 40      # pre-event volume baseline for volume_decay
POST_WINDOW_DAYS = 20     # post-event median close and volume

#: Forward horizon for the outcome label (SEC suspension / delisting /
#: enforcement action within this many CALENDAR days of the event).
LABEL_HORIZON_DAYS = 180

#: Candidate-event detection.
#:
#: Feature 1 as specified (window $vol / trailing 2y $vol) has a mechanical
#: floor of window_len / trailing_days: with only 60 days of history a 5-day
#: window is 8.3% of it even when volume is perfectly flat. Thresholding the
#: raw ratio therefore manufactures candidates for every newly-listed ticker.
#: Detection instead uses the normalised form -- the window's share divided by
#: its proportional share -- which is 1.0 under flat volume regardless of how
#: much history exists. A value of 5 means the window carried five times its
#: proportional share of trailing liquidity.
CANDIDATE_MIN_CONCENTRATION_RATIO = 5.0
#: Retained because feature 1 is reported verbatim as specified.
CANDIDATE_MIN_CONCENTRATION = 0.05
#: Minimum trailing history before any concentration statistic is meaningful.
MIN_TRAILING_DAYS = 120
#: Minimum trading days between two accepted anchors for one ticker, so a
#: single episode yields one event rather than a smear of overlapping windows.
EVENT_REFRACTORY_DAYS = 30

#: Small-cap universe filter applied at event time (point-in-time, never from
#: a current-day snapshot).
MAX_MARKET_CAP = 2_000_000_000.0
MIN_PRICE = 0.10

#: Data-integrity thresholds.
#: A close-to-close ratio at or beyond this that reverses next day and is not
#: matched by a known corporate action is treated as a suspected unadjusted
#: split. Reverse splits fake the run-up signature exactly, so this check is
#: mandatory rather than advisory.
SPLIT_SUSPECT_RATIO = 1.75
#: Minimum share of the universe that must be terminated securities before the
#: panel is considered free of survivorship bias.
MIN_DEAD_TICKER_SHARE = 0.05
#: Share of tickers with a suspected unadjusted split above which the feed is
#: treated as systematically unadjusted (fatal) rather than as heuristic noise
#: (warning + per-event flag). A genuinely unadjusted feed trips on every
#: split in the panel, not on a handful.
MAX_SUSPECT_SPLIT_TICKER_SHARE = 0.02


@dataclass(frozen=True)
class Paths:
    """Output locations for the deliverables."""

    root: Path = Path(__file__).resolve().parents[1] / "artifacts"

    @property
    def feature_store(self) -> Path:
        return self.root / "features.parquet"

    @property
    def model(self) -> Path:
        return self.root / "model.joblib"

    @property
    def calibration_plot(self) -> Path:
        return self.root / "calibration.png"

    @property
    def screen(self) -> Path:
        return self.root / "screen.csv"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation.json"

    @property
    def integrity(self) -> Path:
        return self.root / "integrity.json"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


@dataclass(frozen=True)
class FeatureSpec:
    """Which feature columns feed the model, and which are diagnostics only."""

    #: Features knowable at the END of the event window. These are the only
    #: ones a live screen may use -- see `retrospective` below.
    causal: tuple[str, ...] = (
        "concentration_ratio",
        "float_turnover",
        "runup",
        "overnight_share",
        "news_filing_in_window",
        "gap_without_news",
        "days_since_ipo_log",
        # Causal variant: a 424B5/S-3/S-1 filed in the 10 days BEFORE the
        # anchor. The spec's `offering_within_10d` looks 10 days forward and
        # is therefore retrospective -- see below.
        "offering_prior_10d",
        "prior_reverse_splits",
    )
    #: Features 4 and 5 of the spec require 20 trading days AFTER the window.
    #: They are legitimate for forensic scoring and for training, but a model
    #: that uses them cannot rank today's events -- it is 20 days late. Keeping
    #: the split explicit prevents that leak from reaching the screen.
    retrospective: tuple[str, ...] = (
        "retracement",
        "volume_decay",
        # Forward-looking by construction: an offering filed after the ramp is
        # not knowable during it. Legitimate for forensic scoring, never for
        # the screen.
        "offering_within_10d",
    )

    @property
    def core(self) -> tuple[str, ...]:
        """Every modelled feature, in spec order.

        `news_filing_in_window` and `gap_without_news` are additions to the
        original list: `overnight_share` is not single-signed on its own
        (scheduled news gaps harder than a ramp; ordinary noise gaps less), so
        it needs news presence alongside it to be usable by a linear model.
        """
        return (
            "concentration_ratio",
            "float_turnover",
            "runup",
            "retracement",
            "volume_decay",
            "overnight_share",
            "news_filing_in_window",
            "gap_without_news",
            "days_since_ipo_log",
            "offering_within_10d",
            "prior_reverse_splits",
        )
    #: Present only when an intraday source is wired in; NaN otherwise and
    #: dropped from the model rather than silently imputed to a wrong value.
    intraday: tuple[str, ...] = (
        "closing_auction_share",
        "small_trade_share",
    )
    #: Carried through the feature store for slicing but never fed to a model.
    diagnostics: tuple[str, ...] = (
        "liquidity_concentration",
        "peak_relative_volume",
        "window_dollar_volume",
        "float_shares",
        "market_cap_proxy",
        "suspected_unadjusted_split",
    )

    def model_columns(self, mode: str = "forensic",
                      include_intraday: bool = False) -> list[str]:
        """Columns for a model.

        mode="screen"    -> causal features only (rankable in real time)
        mode="forensic"  -> causal + retrospective (needs 20 days of hindsight)
        """
        if mode == "screen":
            cols = list(self.causal)
        elif mode == "forensic":
            cols = list(self.causal) + list(self.retrospective)
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        if include_intraday:
            cols += list(self.intraday)
        return cols


FEATURES = FeatureSpec()
