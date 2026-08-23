"""Command line entry points."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import EVENT_WINDOWS, PATHS, PRIMARY_WINDOW


def _add_source_args(p, default_source="synthetic"):
    p.add_argument("--source", default=default_source,
                   choices=["synthetic", "csv", "polygon", "yfinance"])
    p.add_argument("--path", default=None, help="data directory (csv source)")
    p.add_argument("--universe", default=None,
                   help="ticker list file, one per line or a CSV with a `ticker` "
                        "column (yfinance source)")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--tickers", type=int, default=1200, help="synthetic universe size")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-shares-history", action="store_true",
                   help="skip yfinance get_shares_full(); much faster, but float "
                        "turnover then has no point-in-time denominator")


def _load_panel(args, parser):
    if args.source == "synthetic":
        from .data.synthetic import SyntheticConfig, generate_panel
        return generate_panel(SyntheticConfig(n_tickers=args.tickers, seed=args.seed))
    if args.source == "csv":
        from .data.csv_source import CsvSource
        if not args.path:
            parser.error("--path is required for --source csv")
        return CsvSource(args.path).load(args.start, args.end)
    if args.source == "yfinance":
        from .data.yfinance_source import YFinanceSource, load_universe_file
        if not args.universe:
            parser.error("--universe is required for --source yfinance")
        tickers = load_universe_file(args.universe)
        src = YFinanceSource(tickers, start=args.start, end=args.end,
                             use_shares_history=not args.no_shares_history)
        panel = src.load()
        if src.missing:
            print(f"note: {len(src.missing)} of {len(tickers)} tickers returned no "
                  f"history (delisted names are the usual cause): "
                  f"{', '.join(src.missing[:12])}"
                  + (" ..." if len(src.missing) > 12 else ""))
        return panel
    from .data.polygon import PolygonSource
    return PolygonSource().load(args.start, args.end)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ldx", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="full supervised pipeline (needs a labelled, "
                                   "survivorship-free panel)")
    _add_source_args(r)
    r.add_argument("--splits", type=int, default=5)
    r.add_argument("--no-strict", action="store_true",
                   help="downgrade fatal integrity findings to warnings")

    s = sub.add_parser("screen", help="unsupervised ranked screen; works without "
                                      "labels, so it suits survivor-only sources")
    _add_source_args(s, default_source="yfinance")
    s.add_argument("--top", type=int, default=50)
    s.add_argument("--asof", default=None)
    s.add_argument("--lookback-days", type=int, default=365)
    s.add_argument("--window", type=int, default=PRIMARY_WINDOW, choices=list(EVENT_WINDOWS))

    a = sub.add_parser("audit", help="integrity audit only")
    _add_source_args(a)

    args = ap.parse_args(argv)

    if args.cmd == "audit":
        from .data.integrity import audit_panel
        rep = audit_panel(_load_panel(args, ap), strict=False)
        print(rep.summary())
        return 0 if rep.ok else 1

    if args.cmd == "screen":
        return _screen(args, ap)

    from .pipeline import run
    run(_load_panel(args, ap), paths=PATHS, n_splits=args.splits,
        strict_integrity=not args.no_strict)
    return 0


def _screen(args, parser) -> int:
    from .data.integrity import audit_panel, detect_unadjusted_splits
    from .features import build_features
    from .unsupervised import rank, score_events

    panel = _load_panel(args, parser)
    PATHS.ensure()

    # Never strict here: a survivor-only source is expected for this command,
    # and the audit's job is to state the consequence, not to block the screen.
    rep = audit_panel(panel, strict=False)
    print(rep.summary())
    rep.save(PATHS.integrity)

    flags = detect_unadjusted_splits(panel.bars, panel.corporate_actions)
    feats = build_features(panel, windows=(args.window,), split_flags=flags)
    if not len(feats):
        print("no candidate events found")
        return 1
    feats.to_parquet(PATHS.feature_store, index=False)
    print(f"\nfeature store -> {PATHS.feature_store}  {feats.shape}")

    scored = score_events(feats)
    print(f"score axes used   : {', '.join(scored.attrs['features_used'])}")
    if scored.attrs["features_missing"]:
        print(f"score axes missing: {', '.join(scored.attrs['features_missing'])}"
              f"  (weights renormalised over the rest)")

    out = rank(feats, top_n=args.top, asof=args.asof, lookback_days=args.lookback_days)
    out.to_csv(PATHS.screen, index=False)
    print(f"ranked screen -> {PATHS.screen}  ({len(out)} rows)\n")
    show = [c for c in ("rank", "ticker", "event_date", "signature_score", "drivers",
                        "float_turnover", "runup", "retracement") if c in out.columns]
    with pd.option_context("display.width", 200, "display.max_colwidth", 46):
        print(out[show].head(20).to_string(index=False))
    print("\nThis ranking is an unvalidated heuristic, not a model: with no labels "
          "there is no precision to quote.\nOn the validation panel roughly 40% of "
          "the top 50 were legitimate confounds -- short squeezes, meme moves and\n"
          "biotech readouts. Treat every row as a signature to check, never a verdict. "
          "See METHODS.md sect. 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
