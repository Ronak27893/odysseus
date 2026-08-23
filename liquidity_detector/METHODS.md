# Methods note

What this estimator measures, what it cannot distinguish, and which numbers
are evidence about markets versus evidence about the code.

## 1. Read this first: the provenance of the current numbers

Every metric currently in `artifacts/` was produced on a **synthetic panel**.
This environment's egress policy returns 403 at the proxy for `sec.gov`,
`data.sec.gov`, `efts.sec.gov` and Yahoo, so no live regulatory or price data
could be fetched. The synthetic source validates the machinery. It is not
evidence about markets.

Concretely, the following are meaningful here:

- feature arithmetic (checked against hand-computable series in
  `tests/test_features.py`)
- that walk-forward splits are forward-only and purged, and that removing the
  embargo demonstrably reintroduces leakage (`tests/test_evaluation.py`)
- that the survivorship and split-adjustment audits fire on panels that
  deserve it, and stay quiet on ones that do not
- that the control false-positive harness reports per-confound rates
- that every deliverable is produced and internally consistent

The following are **not** meaningful here and must not be quoted:

- coefficient magnitudes and signs
- PR-AUC, precision@k, lift
- the control false-positive rate

Those are readbacks of generator parameters. The leak detector makes this
visible: it flags `volume_decay` at AUC 0.96 and `float_turnover` at 0.87,
which is not a discovery — it is the generator's `volume_decay` and
`window_float_turnover` ranges being recovered.

## 2. What the model cannot distinguish

This is the section that matters, and no amount of additional data fixes most
of it.

**Manufactured exit vs. short squeeze.** Both compress enormous volume into a
few days, both trade multiples of the float, both round-trip. On the
validation panel short squeezes are the most-confused control. Separating them
requires short interest, borrow rates and days-to-cover — none of which are in
this feature set.

**Manufactured exit vs. attention shock.** A genuinely news-driven crowd
produces the same price/volume geometry as an engineered one. This is not a
modelling deficiency; the two are observationally equivalent in price and
volume, which is exactly why an attention-driven large cap can rank alongside
a shell. Any flag on a widely-covered name should be read as "this name's
liquidity profile changed shape", never as "this name was manipulated".

**Manufactured exit vs. legal dilution into strength.** A biotech that gaps on
a positive readout and files a 424B5 four days later has done something
entirely lawful and looks nearly identical to a dilution-into-strength scheme.
The features cannot separate lawful opportunism from an engineered exit,
because the observable conduct is the same. Only the intent differs, and
intent is not in the data.

**Who actually sold.** The hypothesis is about a *position* being exited. No
feature here observes holders. Float turnover is a proxy — it establishes that
enough volume existed for a large holder to get out, never that one did. The
data that would actually test the hypothesis is holder-level: SC 13D/G exits,
Form 4 insider sales, 13F position deltas, S-1 selling-shareholder tables.
Until that layer exists, this pipeline tests a *necessary* condition, not the
hypothesis itself.

**Intent, coordination, or promotion.** Nothing here observes stock promoters,
paid campaigns, boiler rooms or coordinated messaging.

## 3. Survivorship bias — the primary failure mode

Suspended issuers get delisted and vendor history disappears. A universe drawn
from currently-listed names has the positive class removed by construction.
The earlier exploratory pass hit exactly this: of 100 SEC suspensions, 11
mapped to tickers and 2 had usable history.

`audit_panel` therefore refuses, fatally, any panel where terminated
securities are under 5% of the universe. This cannot be satisfied by a
screener export. It requires CRSP, Polygon's `active=false` reference set, or
an equivalent point-in-time source that retains dead tickers.

## 4. Label impurity and censoring

**The negative class contains undetected manipulation.** Enforcement is a
lagged, filtered, resource-constrained observation of the phenomenon, not a
census. Reported precision is a **lower bound**; reported recall is against
*detected* cases only. `censoring_report` restates this on every run.

**Form 25 is ambiguous.** A delisting notice is filed for completed mergers
and going-private transactions as often as for deficiency. `build_label_events`
will not promote delistings to positives unless given an explicit reason
mapping — without it, only SEC suspensions are clean positives.

**Recent events are censored, not negative.** An event whose 180-day horizon
extends past the data's end has an unobservable outcome. Labelling it 0 would
teach the model that recent events are safe. Those rows are excluded from
training and counted in the report.

**Enforcement lag is not modelled.** The 180-day horizon is a choice, not a
finding. A suspension can follow the event by years.

## 5. Temporal leakage

Two leaks were found by tests during development, both of which would have
inflated results:

1. **`offering_within_10d` in the screen model.** It looks 10 days *forward*
   of the anchor. `tests/test_causality.py` caught it; the causal variant
   `offering_prior_10d` replaced it in the screen feature set.
2. **Unpurged fold boundaries.** With a 180-day forward label, a training
   event near a fold boundary has its outcome revealed inside the test window.
   Folds are purged by the full horizon;
   `test_zero_embargo_would_leak` confirms the guard actually bites.

## 6. Feature-construction findings

**Feature 1 is contaminated by history length.** `window $vol / trailing 2y
$vol` has a mechanical floor of `window_len / trailing_days`. At 60 days of
history a 5-day window is 8.3% of trailing volume *under perfectly flat
volume*. Thresholding it manufactures a candidate event for every newly-listed
ticker. Detection uses the normalised ratio instead.

**Conditioning removes the screening variable's power.** Because candidates
are selected on concentration, concentration barely discriminates *within* the
candidate set (AUC ≈ 0.58). This is expected and is a reason not to read a
near-zero coefficient on feature 1 as "concentration doesn't matter".

**`overnight_share` is not single-signed.** This was the feature flagged as
most likely to add precision, and the validation run says the original framing
is half right. Median overnight share by group:

| group | n | median `overnight_share` |
|---|---|---|
| M&A announcement | 29 | 0.90 |
| earnings surprise | 41 | 0.87 |
| biotech readout | 30 | 0.82 |
| **manufactured** | **45** | **0.71** |
| short squeeze | 28 | 0.51 |
| ordinary noise | 820 | 0.45 |
| meme attention | 26 | 0.41 |
| index inclusion | 42 | 0.32 |

Ramps gap *more* than ordinary volume spikes and attention-driven moves, and
*less* than scheduled corporate news — because material news is released
outside market hours, which is a fact about disclosure practice rather than
about any generator. So the feature sits in the middle of the distribution and
a single monotone coefficient cannot use it.

Pairing it with news presence fixes this. US issuers must file an 8-K within
four business days of a material corporate event, so "gapped, and nothing was
filed" is the discriminating cell. Ablated on one panel with identical folds,
adding the news pair moved the screen model's PR-AUC from **0.602 to 0.660**
and roughly halved the control false-positive rate (**0.032 to 0.013**).
`news_filing_in_window` became the largest single coefficient — odds ratio
0.17 per SD, i.e. a filed 8-K makes an event markedly less suspicious — and
`overnight_share` turned positive once conditioned on it.

The explicit product term `gap_without_news` added nothing beyond the two main
effects (coefficient -0.16), which is expected: a linear model reconstructs
the interaction from them.

**This structure should transfer to real data; the magnitudes should not be
assumed to.** The mechanism behind it — scheduled news is disclosed outside
market hours and leaves a filing — is a fact about disclosure practice, not a
property of the generator.

## 7. Scores are ranks, not probabilities

`class_weight="balanced"` reweights the positive class, so predicted
probabilities are inflated. Measured out-of-fold on the validation panel: mean
predicted 0.124 against an observed rate of 0.057 — roughly 2x over-confident,
ECE 0.068 for logistic, 0.019–0.024 for the GBM. The reliability curve sits
below the diagonal throughout.

Use the output to **rank**. To read it as a probability, fit a calibrator
(isotonic or Platt) on a held-out tail of each training fold — never on the
test fold.

## 8. Float data quality

Float turnover above roughly 20x should be treated as a **data-quality flag
first and a signal second**. A stale `floatShares` snapshot from a vendor
produces exactly that reading. `audit_panel` warns when float history has one
record per ticker, which means it is a snapshot rather than point-in-time, and
`float_as_of` reads the most recent record at or before the event date so a
later offering cannot backdate share count.

## 9. Statistical power

The base rate is 1–6%. Positive counts are in the tens even across a
multi-thousand-name decade-long panel. Consequently:

- metrics are pooled out-of-fold, because per-fold estimates on 5–10 positives
  are meaningless
- every PR-AUC carries a percentile bootstrap CI, and those intervals are
  wide — that width is the finding, not a presentation problem
- ROC-AUC is reported only to show how much it flatters at this base rate
  (0.94–0.97 against PR-AUC 0.61–0.72)

## 10. The yfinance path specifically

yfinance is reachable and free, and it is the right tool for exactly one of
the two jobs here.

**What it supports.** The unsupervised screen. Candidate detection, every
price/volume feature, the split-adjustment audit and the ranking all work on
live names.

**What it cannot support.** Supervised training, validation, or any quoted
performance number. Yahoo drops history for suspended and delisted issuers, so
the positive class is absent by construction — not scarce, absent. The audit
fails a yfinance panel deliberately and says so; running the supervised
pipeline anyway would fit a model against a negative class and report metrics
that mean nothing. This is the same wall the earlier exploratory pass hit when
only 2 of 100 suspensions had usable price history, and no amount of care in
the modelling fixes it. It is a property of the data source.

**Shares outstanding is not float.** `get_shares_full()` returns shares
outstanding. Float is smaller — often much smaller for a recent IPO or a
controlled company — so `float_turnover` computed on a yfinance panel is a
**lower bound**. Read a value of 3x as "at least three times the float", which
is still the interesting statement, but do not treat the number as the float
multiple. The panel records `share_count_basis` so this is not lost downstream.

**Use `Close`, never `Adj Close`.** `Close` is split-adjusted and `Volume` is
split-adjusted to match. `Adj Close` is additionally dividend-adjusted, which
misstates both the traded price level and dollar volume. The adapter is
pinned to `auto_adjust=False` and a test asserts which column is read, because
this is a silent-wrong-answer failure rather than a crash.

**The score is a prior, not a model.** Weights in `ldx/unsupervised.py` follow
the hypothesis — float turnover heaviest, because the claim is that the move
exists to create exit depth — and were not tuned against any outcome. With no
labels there is no precision to report and no calibration to check.

**Confound contamination is the binding limitation, and it is large.** Scored
against the synthetic panel, where labels do exist, the heuristic ranks well
against random names (precision@10 of 0.70, roughly 16x the base rate) — but
**19 of the top 50 were legitimate confounds**: 11 short squeezes, 5
meme-attention moves, 3 biotech readouts. That is the SPCE problem, quantified.
A ~40% confound rate in the top of the ranking is the realistic expectation for
any price-and-volume-only screen, and it is why the EDGAR layer matters: adding
filing presence was worth more than any price feature (§6).

## 11. What would actually advance this

In descending order of expected value:

1. **Holder-level data.** SC 13D/G exit filings, Form 4, 13F deltas, S-1
   selling-shareholder tables. This is the only way to test the actual
   hypothesis rather than a necessary condition of it.
2. **A point-in-time panel including delisted names** (CRSP or Polygon flat
   files with `active=false`). Without it there is no positive class.
3. **Short interest and borrow data**, to separate the squeeze control.
4. **Intraday/TAQ**, for the closing-auction share and trade-size distribution
   the current pipeline leaves NaN when no intraday source is wired in.
5. **A curated suspension label set.** The SEC suspension page layout has
   changed repeatedly; `parse_suspension_table` is best-effort and a scraper
   that silently returns nothing is worse than a reviewed CSV of a few hundred
   rows.
