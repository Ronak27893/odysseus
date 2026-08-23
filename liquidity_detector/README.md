# Manufactured-liquidity-event detector

Screens US small caps for events where a price move looks like it existed to
create enough depth for someone to get out, rather than to reprice the
security.

**Hypothesis under test.** In ramp-and-dump and dilution-into-strength events
the price move is not the objective — it is the mechanism for creating enough
depth to exit a position that could not otherwise be sold. The operative
quantity is therefore not the run-up but the share of the float that changed
hands while the run-up made that possible.

Every flag is a signature, not a verdict. Read `METHODS.md` before acting on
any output; it states what the model provably cannot distinguish.

## Status

The pipeline is complete and tested end to end. **The numbers it currently
produces come from a synthetic panel, not from markets.** This environment's
egress policy blocks `sec.gov`, `data.sec.gov`, `efts.sec.gov` and Yahoo
(403 at the proxy), so no live regulatory or price data could be fetched here.
The synthetic source exists to validate the machinery — feature arithmetic,
purged walk-forward splitting, calibration, precision@k, the control
false-positive harness — not to make claims about real markets. Point it at
real data via the adapters below and the same commands produce real results.

## Install and run

```bash
pip install -r requirements.txt
python -m ldx.cli run --tickers 1200          # synthetic validation run (supervised)
python -m ldx.cli screen --source yfinance --universe universe.txt  # unsupervised
python -m ldx.cli audit --source csv --path /data/crsp   # integrity only
python -m pytest tests/ -q                    # 71 tests
```

## Data adapters

| Source | Use |
|---|---|
| `ldx.data.csv_source.CsvSource` | **Recommended.** A directory of CSV/Parquet — CRSP extract, Nasdaq Data Link, Polygon flat files. |
| `ldx.data.polygon.PolygonSource` | Polygon REST. Calls `/v3/reference/tickers?active=false` so delisted names are included. Needs `POLYGON_API_KEY`. |
| `ldx.data.yfinance_source.YFinanceSource` | Free and easy, **survivor-only**. Good for the unsupervised screen; cannot support supervised training — see below. |
| `ldx.data.synthetic` | Offline validation only. |
| `ldx.edgar.EdgarClient` | Filing history, full-text search, suspension list. Needs `SEC_USER_AGENT` with contact info; rate-limited to 8 req/s. |

`CsvSource` expects `bars` and `securities`; `float_history`,
`corporate_actions`, `filings`, `label_events`, `control_events` and
`intraday` are optional. Column contracts are in `ldx/data/csv_source.py`.

Two requirements no loader can verify for you, so `audit_panel` measures them
from the bars: **prices must be split-adjusted** and **the universe must
include delisted securities**.

## Using yfinance

```bash
printf 'ABCD\nEFGH\nIJKL\n' > universe.txt
python -m ldx.cli screen --source yfinance --universe universe.txt \
    --start 2019-01-01 --top 50
```

`screen` is the unsupervised path: it ranks candidate events by how far they
sit from the candidate population on the hypothesised axes, so it needs no
labels. That matters because **yfinance cannot supply the positive class** —
Yahoo drops history for suspended and delisted issuers, which is precisely the
population the labels come from. `ldx.cli run` (supervised) will fail the
integrity audit on a yfinance panel, by design, with a message saying why.

Three details the adapter gets right, each of which caused a problem in the
earlier exploratory pass:

- **`Close`, not `Adj Close`.** Yahoo's `Close` is split-adjusted and its
  `Volume` is split-adjusted to match — exactly what this model needs.
  `Adj Close` is additionally dividend-adjusted, which distorts both the traded
  price level and dollar volume. The adapter passes `auto_adjust=False` and
  reads `Close`; `tests/test_yfinance_source.py` proves which column is used.
- **Share counts come from `get_shares_full()`, a time series** — not the
  `floatShares` snapshot in `.info`. A snapshot is the wrong denominator for an
  event three years ago and is the most likely source of implausible readings
  like 124x turnover. Pass `--no-shares-history` to skip it (much faster, but
  then there is no point-in-time denominator).
- **Shares outstanding is not float.** yfinance's series is shares
  outstanding, which is larger than float, so turnover computed against it is a
  **lower bound** on true float turnover. The adapter records the basis on the
  panel rather than passing one off as the other.

Reverse splits are read from `Ticker.splits` and normalised (yfinance reports
a 1-for-10 as `0.1`), so declared splits are excluded from the unadjusted-split
detector instead of firing it.

## What runs, in order

1. **Integrity audit** (`ldx/data/integrity.py`). Refuses a survivor-only
   universe. Detects unadjusted splits from the data — a price jump near a
   common split ratio, persisting, with volume moving *inversely*. That last
   test is what separates a 1-for-10 reverse split from a genuine ramp; both
   look identical on price alone.
2. **Candidate detection + features** (`ldx/features.py`). One row per
   (ticker, event_date, window) for windows 3/5/10 anchored on the *same*
   event.
3. **Labelling** (`ldx/labeling.py`). Termination within 180 days, with
   censoring and control tagging.
4. **Leak check** (`univariate_diagnostics`). Any feature separating at
   AUC ≥ 0.95 on its own is flagged before a model is fitted.
5. **Models** (`ldx/model.py`). Logistic regression first, then gradient
   boosting — in that order, so the coefficients are readable.
6. **Evaluation** (`ldx/evaluation.py`). Purged walk-forward, PR-AUC with
   bootstrap CIs, precision@k, control FPR by confound type, ECE.
7. **Deliverables** → `artifacts/`: `features.parquet`, `model.joblib`,
   `calibration.png`, `screen.csv`, `evaluation.json`, `integrity.json`.

For sources with no positive class, `ldx/unsupervised.py` replaces steps 3–6
with a transparent weighted robust-z composite. It is a stated prior, not a
fitted model, and nothing about it is validated — see `METHODS.md` §10.

## Screen vs forensic

The spec's feature list mixes two regimes, and conflating them silently
backdates hindsight into a live ranking:

- **Screen** — features knowable at the window's end. This is what can rank
  today's events.
- **Forensic** — adds `retracement` and `volume_decay` (20 trading days of
  hindsight) and `offering_within_10d` (looks 10 days forward).

`FeatureSpec.model_columns("screen"|"forensic")` enforces the split and
`tests/test_causality.py` guards it. On the validation panel the screen model
gives up roughly 8% of PR-AUC relative to forensic (0.660 vs 0.715) — the
honest price of ranking in real time.

## Deviations from the original spec

Three, each for a reason found while testing:

1. **Feature 1 is normalised.** `window $vol / trailing 2y $vol` has a
   mechanical floor of `window_len / trailing_days`: with 60 days of history a
   5-day window is 8.3% of it even under flat volume, so thresholding the raw
   ratio manufactures candidates for every newly-listed ticker. Detection uses
   `concentration_ratio` (share ÷ proportional share, = 1.0 under flat volume
   at any history length). The raw feature is still computed and reported.
2. **`offering_within_10d` was split.** It looks forward, so a causal
   `offering_prior_10d` was added for the screen.
3. **`overnight_share` gained a companion.** It is not single-signed — see
   `METHODS.md`. `news_filing_in_window` was added alongside it.
