"""Walk-forward evaluation, precision@k, PR-AUC, calibration, control FPR.

Three choices here are not stylistic:

1. Splits are by calendar time and PURGED. The label looks forward
   `LABEL_HORIZON_DAYS`; a training event whose outcome is only revealed after
   the test window opens leaks. Those rows are dropped from the training side
   of each fold rather than merely ordered.
2. Metrics are PR-AUC and precision@k. At a ~2% base rate, ROC-AUC flatters
   and accuracy is meaningless.
3. Point estimates come with bootstrap intervals. With tens of positives, a
   PR-AUC quoted to three decimals without an interval is false precision.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import LABEL_HORIZON_DAYS


@dataclass
class Fold:
    index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_train_pos: int
    n_test: int
    n_test_pos: int
    n_purged: int


def walk_forward_folds(dates: pd.Series, n_splits: int = 5,
                       min_train_frac: float = 0.35,
                       embargo_days: int = LABEL_HORIZON_DAYS
                       ) -> list[tuple[np.ndarray, np.ndarray, Fold]]:
    """Expanding-window folds over calendar time, purged by the label horizon."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    order = np.argsort(d.to_numpy())
    d_sorted = d.iloc[order]
    n = len(d)
    start_cut = int(n * min_train_frac)
    if start_cut < 1 or n_splits < 1:
        return []
    cuts = np.linspace(start_cut, n, n_splits + 1).astype(int)
    folds = []
    for i in range(n_splits):
        te_lo, te_hi = cuts[i], cuts[i + 1]
        if te_hi - te_lo < 1:
            continue
        test_idx = order[te_lo:te_hi]
        test_start = d.iloc[test_idx].min()
        train_pool = order[:te_lo]
        # Purge: drop training rows whose outcome window reaches into the test
        # period -- their label is not knowable at the fold boundary.
        keep = d.iloc[train_pool] + pd.Timedelta(days=embargo_days) <= test_start
        train_idx = train_pool[keep.to_numpy()]
        n_purged = int(len(train_pool) - len(train_idx))
        if len(train_idx) < 30:
            continue
        folds.append((train_idx, test_idx, Fold(
            index=i,
            train_start=str(d.iloc[train_idx].min().date()),
            train_end=str(d.iloc[train_idx].max().date()),
            test_start=str(test_start.date()),
            test_end=str(d.iloc[test_idx].max().date()),
            n_train=len(train_idx), n_train_pos=0,
            n_test=len(test_idx), n_test_pos=0, n_purged=n_purged,
        )))
    return folds


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    if k <= 0 or len(y) == 0:
        return float("nan")
    k = min(k, len(y))
    top = np.argsort(-score)[:k]
    return float(y[top].mean())


def bootstrap_ci(y: np.ndarray, score: np.ndarray, stat, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap over events. Wide intervals here are the point."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0 or y.sum() == 0:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        try:
            vals.append(stat(y[idx], score[idx]))
        except ValueError:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


@dataclass
class EvalResult:
    model: str
    mode: str
    n_events: int
    n_positive: int
    base_rate: float
    pr_auc: float
    pr_auc_ci: tuple[float, float]
    pr_auc_lift: float
    roc_auc: float
    precision_at_k: dict[str, float] = field(default_factory=dict)
    lift_at_k: dict[str, float] = field(default_factory=dict)
    control_flag_rate: dict[str, float] = field(default_factory=dict)
    control_flag_rate_overall: float = float("nan")
    #: Expected calibration error, and the direction of the miscalibration.
    ece: float = float("nan")
    mean_predicted: float = float("nan")
    observed_rate: float = float("nan")
    folds: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        ci = self.pr_auc_ci
        lines = [
            f"[{self.model}/{self.mode}] n={self.n_events} pos={self.n_positive} "
            f"base={self.base_rate:.3%}",
            f"  PR-AUC {self.pr_auc:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  "
            f"lift x{self.pr_auc_lift:.1f}   (ROC-AUC {self.roc_auc:.3f}, reported "
            f"only to show why it flatters)",
        ]
        for k, v in self.precision_at_k.items():
            lines.append(f"  precision@{k:<5s} {v:.3f}   lift x{self.lift_at_k.get(k, float('nan')):.1f}")
        if self.control_flag_rate:
            lines.append(f"  control flag rate (overall {self.control_flag_rate_overall:.3f}):")
            for k, v in sorted(self.control_flag_rate.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {k:<20s} {v:.3f}")
        if np.isfinite(self.ece):
            lines.append(f"  calibration: ECE {self.ece:.3f}  mean predicted "
                         f"{self.mean_predicted:.3f} vs observed {self.observed_rate:.3f}"
                         + ("  [OVER-confident]" if self.mean_predicted > self.observed_rate
                            else "  [UNDER-confident]"))
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def evaluate_walk_forward(df: pd.DataFrame, columns: list[str], model_factory,
                          model_name: str, mode: str, n_splits: int = 5,
                          k_list: tuple[int, ...] = (10, 25, 50),
                          seed: int = 0) -> tuple[EvalResult, pd.DataFrame]:
    """Fit and score out-of-fold, then measure. Returns (result, oof frame)."""
    df = df.sort_values("event_date").reset_index(drop=True)
    y_all = df["label"].to_numpy(dtype=int)
    folds = walk_forward_folds(df["event_date"], n_splits=n_splits)
    oof = np.full(len(df), np.nan)
    fold_meta = []
    for train_idx, test_idx, meta in folds:
        ytr = y_all[train_idx]
        meta.n_train_pos = int(ytr.sum())
        meta.n_test_pos = int(y_all[test_idx].sum())
        fold_meta.append(asdict(meta))
        if meta.n_train_pos < 3:
            continue  # cannot fit a class-balanced model on <3 positives
        pipe = model_factory(columns)
        pipe.fit(df.iloc[train_idx][columns + []], ytr)
        oof[test_idx] = pipe.predict_proba(df.iloc[test_idx][columns])[:, 1]

    scored = np.isfinite(oof)
    y = y_all[scored]
    s = oof[scored]
    res = EvalResult(
        model=model_name, mode=mode, n_events=int(scored.sum()),
        n_positive=int(y.sum()),
        base_rate=float(y.mean()) if len(y) else float("nan"),
        pr_auc=float("nan"), pr_auc_ci=(float("nan"), float("nan")),
        pr_auc_lift=float("nan"), roc_auc=float("nan"), folds=fold_meta,
    )
    if len(y) and y.sum() > 0:
        res.pr_auc = float(average_precision_score(y, s))
        res.pr_auc_ci = bootstrap_ci(y, s, average_precision_score, seed=seed)
        res.pr_auc_lift = res.pr_auc / res.base_rate if res.base_rate > 0 else float("nan")
        try:
            res.roc_auc = float(roc_auc_score(y, s))
        except ValueError:
            pass
        for k in k_list:
            p = precision_at_k(y, s, k)
            res.precision_at_k[str(k)] = p
            res.lift_at_k[str(k)] = p / res.base_rate if res.base_rate > 0 else float("nan")
        for pct in (0.01, 0.05):
            k = max(1, int(round(pct * len(y))))
            p = precision_at_k(y, s, k)
            res.precision_at_k[f"top{int(pct*100)}%"] = p
            res.lift_at_k[f"top{int(pct*100)}%"] = (p / res.base_rate
                                                    if res.base_rate > 0 else float("nan"))

    # --- control false-positive rate --------------------------------------
    # Threshold set where the screen would actually operate: the top 5% of
    # scores. A control that lands above it is a false positive of the kind
    # price/volume alone cannot rule out.
    sub = df.loc[scored].copy()
    sub["score"] = s
    if len(sub) and y.sum() > 0:
        thr = float(np.quantile(s, 0.95))
        ctl = sub[sub["is_control"]]
        if len(ctl):
            res.control_flag_rate_overall = float((ctl["score"] >= thr).mean())
            for kind, gg in ctl.groupby("control_kind"):
                res.control_flag_rate[str(kind)] = float((gg["score"] >= thr).mean())

    if len(y) and y.sum() > 0:
        res.ece = expected_calibration_error(y, s)
        res.mean_predicted = float(s.mean())
        res.observed_rate = float(y.mean())
        if res.mean_predicted > res.observed_rate * 1.5:
            res.notes.append(
                "scores are OVER-confident: class_weight='balanced' reweights the "
                "positive class, so outputs are valid for RANKING but must not be "
                "read as probabilities without a separate calibration step")

    n_pos_folds = sum(1 for f in fold_meta if f["n_test_pos"] == 0)
    if n_pos_folds:
        res.notes.append(f"{n_pos_folds} of {len(fold_meta)} test folds contain zero "
                         f"positives; per-fold metrics would be undefined, so metrics "
                         f"are pooled out-of-fold")
    res.notes.append(f"training rows purged at fold boundaries to respect the "
                     f"{LABEL_HORIZON_DAYS}-day label horizon: "
                     f"{sum(f['n_purged'] for f in fold_meta)}")
    return res, sub


def expected_calibration_error(y: np.ndarray, s: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean |predicted - observed| across quantile bins."""
    if len(y) == 0:
        return float("nan")
    qs = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        return float("nan")
    idx = np.clip(np.digitize(s, qs[1:-1]), 0, len(qs) - 2)
    total, err = 0, 0.0
    for b in range(len(qs) - 1):
        m = idx == b
        if not m.any():
            continue
        err += m.sum() * abs(s[m].mean() - y[m].mean())
        total += m.sum()
    return float(err / total) if total else float("nan")


def calibration_plot(oof: pd.DataFrame, path, n_bins: int = 8,
                     title: str = "Calibration (out-of-fold)") -> None:
    """Reliability diagram. Written to `path` as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = oof["score"].to_numpy()
    y = oof["label"].to_numpy(dtype=float)
    qs = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        qs = np.linspace(s.min(), s.max() + 1e-9, 4)
    idx = np.clip(np.digitize(s, qs[1:-1]), 0, len(qs) - 2)
    xs, ys, ns = [], [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        xs.append(s[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    lim = max(max(xs + ys) if xs else 1.0, 1e-3) * 1.15
    ax[0].plot([0, lim], [0, lim], "--", color="0.6", lw=1, label="perfect")
    ax[0].plot(xs, ys, "o-", color="#2b6cb0", label="observed")
    for i, (x, yv, nn) in enumerate(zip(xs, ys, ns)):
        ax[0].annotate(f"n={nn}", (x, yv), textcoords="offset points",
                       xytext=(6, 6 if i % 2 == 0 else -12), fontsize=7, color="0.35")
    ece = expected_calibration_error(y, s)
    ax[0].text(0.03, 0.94, f"ECE {ece:.3f}\nbelow the line = over-confident",
               transform=ax[0].transAxes, fontsize=8, color="0.3", va="top")
    ax[0].set_xlabel("mean predicted probability")
    ax[0].set_ylabel("observed frequency")
    ax[0].set_title(title)
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].hist(s, bins=40, color="#2b6cb0", alpha=0.85)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("predicted probability")
    ax[1].set_ylabel("events (log)")
    ax[1].set_title("Score distribution")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_evaluation(results: list[EvalResult], path, extra: dict | None = None) -> None:
    payload = {"results": [r.to_dict() for r in results]}
    if extra:
        payload |= extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


#: Univariate AUC above this is treated as a suspected label leak rather than
#: a strong feature. On real data the usual cause is a feature computed with
#: information from after the label date; on synthetic data it means the
#: generator encoded the label.
LEAK_AUC_THRESHOLD = 0.95


def univariate_diagnostics(df: pd.DataFrame, columns: list[str],
                           label_col: str = "label") -> pd.DataFrame:
    """Per-feature univariate separation, with a leak flag.

    Also reports Mann-Whitney p so a reader can see which single features
    separate at all, independent of the fitted model.
    """
    from scipy.stats import mannwhitneyu

    y = df[label_col].to_numpy(dtype=int)
    rows = []
    for c in columns:
        x = df[c].to_numpy(dtype=float)
        m = np.isfinite(x)
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        pos, neg = x[m & (y == 1)], x[m & (y == 0)]
        if len(pos) < 3 or len(neg) < 3:
            continue
        try:
            auc = float(roc_auc_score(y[m], x[m]))
        except ValueError:
            continue
        try:
            p = float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
        rows.append({
            "feature": c,
            "auc": auc,
            "auc_directional": max(auc, 1 - auc),
            "median_positive": float(np.median(pos)),
            "median_negative": float(np.median(neg)),
            "mannwhitney_p": p,
            "coverage": float(m.mean()),
            "suspected_leak": bool(max(auc, 1 - auc) >= LEAK_AUC_THRESHOLD),
        })
    out = pd.DataFrame(rows)
    return (out.sort_values("auc_directional", ascending=False).reset_index(drop=True)
            if len(out) else out)
