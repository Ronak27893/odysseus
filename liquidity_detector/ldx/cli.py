"""Command line entry points."""
from __future__ import annotations

import argparse
import sys

from .config import PATHS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ldx", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="full pipeline: features, models, deliverables")
    r.add_argument("--source", default="synthetic",
                   choices=["synthetic", "csv", "polygon"],
                   help="panel source; 'synthetic' validates the pipeline offline")
    r.add_argument("--path", default=None, help="data directory (csv source)")
    r.add_argument("--tickers", type=int, default=1200, help="synthetic universe size")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--splits", type=int, default=5)
    r.add_argument("--no-strict", action="store_true",
                   help="downgrade fatal integrity findings to warnings")

    a = sub.add_parser("audit", help="integrity audit only")
    a.add_argument("--source", default="synthetic", choices=["synthetic", "csv", "polygon"])
    a.add_argument("--path", default=None)

    args = ap.parse_args(argv)

    def load_panel():
        if args.source == "synthetic":
            from .data.synthetic import SyntheticConfig, generate_panel
            cfg = SyntheticConfig(n_tickers=getattr(args, "tickers", 1200),
                                  seed=getattr(args, "seed", 7))
            return generate_panel(cfg)
        if args.source == "csv":
            from .data.csv_source import CsvSource
            if not args.path:
                ap.error("--path is required for --source csv")
            return CsvSource(args.path).load(None, None)
        from .data.polygon import PolygonSource
        return PolygonSource().load(None, None)

    if args.cmd == "audit":
        from .data.integrity import audit_panel
        rep = audit_panel(load_panel(), strict=False)
        print(rep.summary())
        return 0 if rep.ok else 1

    from .pipeline import run
    run(load_panel(), paths=PATHS, n_splits=args.splits,
        strict_integrity=not args.no_strict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
