"""Seeded synthetic panel used to validate the pipeline end to end.

This exists because the machinery -- feature definitions, walk-forward
splitting, calibration, precision@k, the control false-positive rate -- has to
be verifiable without a market-data licence. It is NOT evidence about real
markets. Numbers produced from this source describe the estimator, not the
world. See METHODS.md.

Design rule: the controls are deliberately made HARD. Meme-attention and
short-squeeze archetypes overlap the manufactured archetype across run-up,
retracement, volume decay and float turnover; biotech readouts gap overnight
and raise into strength exactly as a dilution event does. A generator tuned to
make the model look good would make the reported control FPR meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import Panel

_BENIGN_DELIST_REASONS = ("merger", "going_private", "acquired")


@dataclass(frozen=True)
class Archetype:
    """Parameters for one event shape. Ranges are (low, high), sampled uniform."""
    name: str
    is_positive: bool
    ramp_days: tuple[int, int]
    peak_multiple: tuple[float, float]
    float_shares: tuple[float, float]
    window_float_turnover: tuple[float, float]
    retracement: tuple[float, float]
    volume_decay: tuple[float, float]
    overnight_share: tuple[float, float]
    offering_prob: float
    reverse_split_lambda: float
    closing_auction_share: tuple[float, float]
    #: Retail-sized-trade share. Elevated for ALL retail-driven episodes
    #: (pumps, memes, squeezes alike) -- deliberately overlapping, because a
    #: microstructure variable that cleanly separated the positive class would
    #: be a label indicator, not a feature.
    small_trade_share: tuple[float, float] = (0.50, 0.62)
    #: Whether material news accompanies the event. US issuers must file an
    #: 8-K within four business days of a material corporate event, so a
    #: readout, a deal or an earnings surprise leaves a filing and a ramp
    #: generally does not. This is what makes "gapped, but nothing was filed"
    #: a meaningful cell rather than noise.
    files_8k_at_event: bool = False
    terminates_prob: float = 0.0


#: The positive class and the six controls named in the spec.
ARCHETYPES: dict[str, Archetype] = {
    "manufactured": Archetype(
        name="manufactured", is_positive=True,
        ramp_days=(3, 8), peak_multiple=(2.0, 7.0),
        float_shares=(1.0e6, 1.6e7), window_float_turnover=(0.8, 14.0),
        retracement=(0.82, 1.0), volume_decay=(0.02, 0.16),
        overnight_share=(0.45, 0.9), offering_prob=0.6,
        reverse_split_lambda=0.9, closing_auction_share=(0.03, 0.12),
        small_trade_share=(0.58, 0.80), terminates_prob=1.0),
    # --- controls: same price/volume shape, different mechanism -----------
    "meme_attention": Archetype(
        name="meme_attention", is_positive=False,
        ramp_days=(3, 10), peak_multiple=(2.0, 8.0),
        float_shares=(8.0e6, 9.0e7), window_float_turnover=(0.5, 6.0),
        retracement=(0.55, 0.95), volume_decay=(0.08, 0.40),
        overnight_share=(0.30, 0.62), offering_prob=0.25,
        reverse_split_lambda=0.1, closing_auction_share=(0.03, 0.10),
        small_trade_share=(0.58, 0.82)),
    "short_squeeze": Archetype(
        name="short_squeeze", is_positive=False,
        ramp_days=(2, 7), peak_multiple=(1.8, 6.0),
        float_shares=(5.0e6, 6.0e7), window_float_turnover=(0.6, 8.0),
        retracement=(0.65, 0.95), volume_decay=(0.10, 0.45),
        overnight_share=(0.32, 0.66), offering_prob=0.20,
        reverse_split_lambda=0.15, closing_auction_share=(0.03, 0.10),
        small_trade_share=(0.55, 0.78)),
    "biotech_readout": Archetype(
        name="biotech_readout", is_positive=False,
        ramp_days=(1, 3), peak_multiple=(1.5, 5.0),
        float_shares=(6.0e6, 7.0e7), window_float_turnover=(0.3, 3.5),
        retracement=(0.10, 0.55), volume_decay=(0.15, 0.50),
        overnight_share=(0.72, 0.95), offering_prob=0.55,
        reverse_split_lambda=0.25, closing_auction_share=(0.04, 0.12),
        small_trade_share=(0.48, 0.70), files_8k_at_event=True),
    "ma_announcement": Archetype(
        name="ma_announcement", is_positive=False,
        ramp_days=(1, 2), peak_multiple=(1.2, 2.6),
        float_shares=(1.0e7, 1.2e8), window_float_turnover=(0.2, 2.0),
        retracement=(0.02, 0.22), volume_decay=(0.20, 0.55),
        overnight_share=(0.85, 0.98), offering_prob=0.02,
        reverse_split_lambda=0.05, closing_auction_share=(0.05, 0.15),
        small_trade_share=(0.42, 0.60), files_8k_at_event=True,
        terminates_prob=0.55),  # deal closes -> benign delisting
    "index_inclusion": Archetype(
        name="index_inclusion", is_positive=False,
        ramp_days=(1, 3), peak_multiple=(1.05, 1.35),
        float_shares=(2.0e7, 1.5e8), window_float_turnover=(0.15, 1.2),
        retracement=(0.30, 0.75), volume_decay=(0.35, 0.75),
        overnight_share=(0.20, 0.50), offering_prob=0.05,
        reverse_split_lambda=0.02, closing_auction_share=(0.35, 0.80),
        small_trade_share=(0.38, 0.55)),
    "earnings_surprise": Archetype(
        name="earnings_surprise", is_positive=False,
        ramp_days=(1, 2), peak_multiple=(1.15, 2.0),
        float_shares=(1.0e7, 1.0e8), window_float_turnover=(0.15, 1.5),
        retracement=(0.20, 0.65), volume_decay=(0.30, 0.65),
        overnight_share=(0.78, 0.96), offering_prob=0.08,
        reverse_split_lambda=0.05, closing_auction_share=(0.04, 0.12),
        small_trade_share=(0.45, 0.62), files_8k_at_event=True),
}


@dataclass
class SyntheticConfig:
    n_tickers: int = 420
    start: str = "2015-01-02"
    end: str = "2024-12-31"
    #: Share of tickers carrying a manufactured (positive) event.
    positive_rate: float = 0.07
    #: Share carrying one of the six controls.
    control_rate: float = 0.24
    #: Benign terminations among event-free names (mergers, going private).
    benign_delist_rate: float = 0.10
    seed: int = 7
    daily_vol: float = 0.045
    base_price: tuple[float, float] = (0.8, 18.0)
    base_volume: tuple[float, float] = (2.0e4, 1.2e6)
    #: Benign volume bursts per ticker per decade, Poisson. EVERY ticker gets
    #: these -- sympathy moves, sector news, index rebalances, guidance. They
    #: are the everyday spikes a screen must reject, and without them the
    #: negative class is unrealistically quiet and the base rate is inflated.
    benign_spikes_lambda: float = 3.5
    benign_spike_volume: tuple[float, float] = (4.0, 30.0)
    benign_spike_move: tuple[float, float] = (0.04, 0.28)


class SyntheticSource:
    """PanelSource implementation backed by the generator above."""

    name = "synthetic"

    def __init__(self, config: SyntheticConfig | None = None):
        self.config = config or SyntheticConfig()

    def load(self, start: str | None = None, end: str | None = None,
             tickers: list[str] | None = None) -> Panel:
        return generate_panel(self.config)


def _u(rng, lo_hi: tuple[float, float]) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def generate_panel(cfg: SyntheticConfig | None = None) -> Panel:
    """Build a complete synthetic Panel, including terminated securities."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    calendar = pd.bdate_range(cfg.start, cfg.end)
    n_days = len(calendar)

    control_names = [k for k, a in ARCHETYPES.items() if not a.is_positive]
    bars_parts, securities, floats, actions, filings = [], [], [], [], []
    labels, controls, intraday_parts = [], [], []

    for i in range(cfg.n_tickers):
        ticker = f"SYN{i:04d}"
        roll = rng.random()
        if roll < cfg.positive_rate:
            arch = ARCHETYPES["manufactured"]
        elif roll < cfg.positive_rate + cfg.control_rate:
            arch = ARCHETYPES[control_names[rng.integers(len(control_names))]]
        else:
            arch = None

        # --- listing lifetime --------------------------------------------
        ipo_idx = int(rng.integers(0, max(1, n_days - 400)))
        ipo_date = calendar[ipo_idx]
        live_len = n_days - ipo_idx

        # --- event anchor -------------------------------------------------
        anchor_idx = None
        if arch is not None and live_len > 300:
            anchor_idx = ipo_idx + int(rng.integers(120, live_len - 120))

        # --- base path ----------------------------------------------------
        p0 = _u(rng, cfg.base_price)
        base_vol_level = _u(rng, cfg.base_volume)
        overnight_frac_base = float(rng.uniform(0.30, 0.55))

        total_ret = rng.normal(0.0, cfg.daily_vol, live_len)
        vol_noise = np.exp(rng.normal(0.0, 0.55, live_len))
        volume = base_vol_level * vol_noise
        # Mild volume autocorrelation.
        for t in range(1, live_len):
            volume[t] = 0.75 * volume[t] + 0.25 * volume[t - 1]

        float_shares = _u(rng, arch.float_shares) if arch else _u(rng, (5.0e6, 9.0e7))
        n_rev_splits = int(rng.poisson(arch.reverse_split_lambda)) if arch else int(rng.poisson(0.06))

        # --- benign volume bursts (all tickers, including clean ones) ------
        n_spikes = int(rng.poisson(cfg.benign_spikes_lambda * live_len / 2520))
        for _ in range(n_spikes):
            si = int(rng.integers(30, max(31, live_len - 25)))
            slen = int(rng.integers(1, 4))
            sl = slice(si, min(live_len, si + slen))
            volume[sl] *= _u(rng, cfg.benign_spike_volume)
            move = _u(rng, cfg.benign_spike_move) * (1 if rng.random() < 0.5 else -1)
            total_ret[sl] += move / max(slen, 1)
            # Partial mean reversion, as ordinary news moves show.
            back = slice(min(live_len, si + slen), min(live_len, si + slen + 5))
            if back.stop > back.start:
                total_ret[back] -= move * float(rng.uniform(0.1, 0.5)) / (back.stop - back.start)

        # --- overlay the event --------------------------------------------
        overnight_share_target = overnight_frac_base
        auction_share_base = float(rng.uniform(0.03, 0.10))
        small_share_base = float(rng.uniform(0.45, 0.60))
        if anchor_idx is not None:
            rel = anchor_idx - ipo_idx
            ramp = int(rng.integers(*arch.ramp_days))
            peak_mult = _u(rng, arch.peak_multiple)
            retrace = _u(rng, arch.retracement)
            decay = _u(rng, arch.volume_decay)
            turnover = _u(rng, arch.window_float_turnover)
            overnight_share_target = _u(rng, arch.overnight_share)
            auction_share_base = _u(rng, arch.closing_auction_share)
            small_share_base = _u(rng, arch.small_trade_share)

            ramp_start = max(1, rel - ramp + 1)
            ramp_slice = slice(ramp_start, rel + 1)
            n_ramp = rel + 1 - ramp_start
            if n_ramp > 0:
                # Spread the log run-up across the ramp with a random shape.
                w = rng.dirichlet(np.ones(n_ramp) * 1.6)
                total_ret[ramp_slice] += np.log(peak_mult) * w
                # Volume for the window is pinned to a float-turnover target.
                window_shares = turnover * float_shares
                vw = rng.dirichlet(np.ones(n_ramp) * 2.0)
                volume[ramp_slice] = window_shares * vw

            # Dump: give back `retrace` of the run-up over the following days.
            dump_len = int(rng.integers(3, 12))
            dump_slice = slice(rel + 1, min(live_len, rel + 1 + dump_len))
            n_dump = max(0, min(live_len, rel + 1 + dump_len) - (rel + 1))
            if n_dump > 0:
                w = rng.dirichlet(np.ones(n_dump) * 1.4)
                total_ret[dump_slice] -= np.log(peak_mult) * retrace * w
                volume[dump_slice] *= np.linspace(0.85, max(decay, 0.02), n_dump) * 6.0
            # Post-event: the market goes away (or does not, for controls).
            post = slice(min(live_len, rel + 1 + dump_len), live_len)
            volume[post] *= decay

        # --- split total return into overnight / intraday -----------------
        gap = total_ret * overnight_share_target
        intra = total_ret - gap
        close = p0 * np.exp(np.cumsum(total_ret))
        close = np.maximum(close, 0.02)
        open_ = np.empty_like(close)
        open_[0] = p0
        open_[1:] = close[:-1] * np.exp(gap[1:])
        open_ = np.maximum(open_, 0.02)
        span = np.abs(rng.normal(0.0, 0.02, live_len)) + 0.004
        high = np.maximum(open_, close) * (1.0 + span)
        low = np.minimum(open_, close) * (1.0 - span)
        volume = np.maximum(np.round(volume), 1.0)

        # --- termination ---------------------------------------------------
        last_idx = live_len - 1
        delist_reason = None
        label_date = None
        if arch is not None and arch.is_positive and anchor_idx is not None:
            lag = int(rng.integers(20, 170))
            end_rel = min(live_len - 1, (anchor_idx - ipo_idx) + lag)
            last_idx = end_rel
            delist_reason = "sec_suspension" if rng.random() < 0.45 else "delisting"
            label_date = calendar[ipo_idx + end_rel]
        elif arch is not None and rng.random() < arch.terminates_prob and anchor_idx is not None:
            lag = int(rng.integers(30, 200))
            end_rel = min(live_len - 1, (anchor_idx - ipo_idx) + lag)
            last_idx = end_rel
            delist_reason = _BENIGN_DELIST_REASONS[rng.integers(len(_BENIGN_DELIST_REASONS))]
        elif arch is None and rng.random() < cfg.benign_delist_rate and live_len > 400:
            last_idx = int(rng.integers(300, live_len))
            delist_reason = _BENIGN_DELIST_REASONS[rng.integers(len(_BENIGN_DELIST_REASONS))]

        keep = slice(0, last_idx + 1)
        dates = calendar[ipo_idx: ipo_idx + last_idx + 1]
        bars_parts.append(pd.DataFrame({
            "ticker": ticker, "date": dates,
            "open": open_[keep], "high": high[keep], "low": low[keep],
            "close": close[keep], "volume": volume[keep],
        }))

        # Intraday microstructure aggregates (auction share, small-trade share).
        n_kept = last_idx + 1
        auction = np.clip(rng.normal(auction_share_base, 0.015, n_kept), 0.005, 0.95)
        small_trade = np.clip(rng.normal(small_share_base, 0.11, n_kept), 0.05, 0.98)
        intraday_parts.append(pd.DataFrame({
            "ticker": ticker, "date": dates,
            "closing_auction_share": auction.astype("float32"),
            "small_trade_share": small_trade.astype("float32"),
        }))

        securities.append({
            "ticker": ticker, "cik": 1_000_000 + i,
            "first_date": dates[0], "last_date": dates[-1],
            "ipo_date": ipo_date,
            "delist_date": dates[-1] if delist_reason else pd.NaT,
            "delist_reason": delist_reason or "",
            "archetype": arch.name if arch else "clean",
        })

        # --- point-in-time float history ----------------------------------
        # Float steps up at each offering; recorded as of the effective date.
        floats.append({"ticker": ticker, "date": dates[0], "float_shares": float_shares})
        n_steps = int(rng.integers(0, 4))
        for s in range(n_steps):
            step_idx = int(rng.integers(1, max(2, len(dates))))
            floats.append({"ticker": ticker, "date": dates[step_idx],
                           "float_shares": float_shares * float(rng.uniform(1.05, 1.9))})

        # --- reverse splits (declared, so the panel stays adjusted) --------
        for s in range(n_rev_splits):
            idx = int(rng.integers(1, max(2, len(dates))))
            actions.append({"ticker": ticker, "date": dates[idx], "kind": "reverse_split",
                            "ratio": float(rng.choice([2, 4, 5, 10, 20]))})

        # --- filings --------------------------------------------------------
        for _ in range(int(rng.integers(1, 5))):
            idx = int(rng.integers(0, len(dates)))
            filings.append({"ticker": ticker, "date": dates[idx],
                            "form": str(rng.choice(["8-K", "S-3", "S-1"]))})
        if anchor_idx is not None and arch.files_8k_at_event:
            # Filed within four business days of the event, as required.
            k_idx = min(len(dates) - 1, (anchor_idx - ipo_idx) + int(rng.integers(0, 4)))
            filings.append({"ticker": ticker, "date": dates[k_idx], "form": "8-K"})
        if anchor_idx is not None and rng.random() < arch.offering_prob:
            off_idx = min(len(dates) - 1, (anchor_idx - ipo_idx) + int(rng.integers(0, 10)))
            filings.append({"ticker": ticker, "date": dates[off_idx], "form": "424B5"})

        if anchor_idx is not None:
            ev_date = calendar[anchor_idx]
            if arch.is_positive:
                labels.append({"ticker": ticker, "date": label_date,
                               "kind": "sec_suspension" if delist_reason == "sec_suspension" else "delisting"})
            else:
                controls.append({"ticker": ticker, "date": ev_date, "kind": arch.name})
        if delist_reason == "delisting" or delist_reason in _BENIGN_DELIST_REASONS:
            filings.append({"ticker": ticker, "date": dates[-1], "form": "25-NSE"})

    panel = Panel(
        bars=pd.concat(bars_parts, ignore_index=True),
        securities=pd.DataFrame(securities),
        float_history=pd.DataFrame(floats),
        corporate_actions=pd.DataFrame(actions, columns=["ticker", "date", "kind", "ratio"]),
        filings=pd.DataFrame(filings),
        label_events=pd.DataFrame(labels, columns=["ticker", "date", "kind"]),
        control_events=pd.DataFrame(controls, columns=["ticker", "date", "kind"]),
        intraday=pd.concat(intraday_parts, ignore_index=True),
        source_name="synthetic",
    )
    return panel
