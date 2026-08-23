"""Turn the regulatory record into label events and filing features.

The positive class is an adverse TERMINATION. Two traps are handled explicitly:

* A Form 25 / 25-NSE is filed for benign reasons too -- completed mergers,
  going-private transactions, voluntary transfers between exchanges. Treating
  every Form 25 as an adverse outcome poisons the label. Callers must pass a
  reason mapping or accept that only SEC suspensions are clean positives.
* EDGAR full-text search only covers filings from 2001 onward, so a panel
  starting earlier has systematically missing filing features rather than
  zeros.
"""
from __future__ import annotations

import re

import pandas as pd

from .client import DELISTING_FORMS, EdgarClient, OFFERING_FORMS

#: Ticker-like token in the suspension list ("...securities of Foo Corp (FOOO)").
_TICKER_RE = re.compile(r"\(([A-Z]{1,5})\)")
_DATE_RE = re.compile(r"(\w+\.?\s+\d{1,2},\s+\d{4})")


def parse_suspension_table(html: str) -> pd.DataFrame:
    """Best-effort parse of the SEC trading-suspension list.

    The page layout has changed several times, so this is intentionally
    forgiving and returns whatever rows it can recognise. For a production run,
    prefer supplying a reviewed CSV of suspensions and skip this entirely --
    the label set is small enough to curate and too important to leave to a
    scraper that silently returns nothing when the markup changes.
    """
    try:
        tables = pd.read_html(html)
    except (ValueError, ImportError):
        tables = []
    rows = []
    for tb in tables:
        cols = {str(c).lower(): c for c in tb.columns}
        date_col = next((cols[c] for c in cols if "date" in c), None)
        name_col = next((cols[c] for c in cols if "company" in c or "name" in c), None)
        if date_col is None or name_col is None:
            continue
        for _, r in tb.iterrows():
            name = str(r[name_col])
            m = _TICKER_RE.search(name)
            rows.append({
                "date": pd.to_datetime(str(r[date_col]), errors="coerce"),
                "company": name,
                "ticker": m.group(1) if m else None,
                "kind": "sec_suspension",
            })
    out = pd.DataFrame(rows, columns=["date", "company", "ticker", "kind"])
    return out.dropna(subset=["date"]).reset_index(drop=True)


def build_filings_frame(client: EdgarClient, tickers: list[str],
                        ticker_cik: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-ticker filing dates for the forms this model uses."""
    if ticker_cik is None:
        ticker_cik = client.ticker_cik_map()
    lookup = dict(zip(ticker_cik["ticker"], ticker_cik["cik"]))
    keep_forms = set(OFFERING_FORMS) | set(DELISTING_FORMS) | {"8-K"}
    parts = []
    for tk in tickers:
        cik = lookup.get(tk)
        if cik is None:
            continue
        try:
            df = client.company_filings(int(cik))
        except Exception:
            continue
        if not len(df):
            continue
        df = df[df["form"].isin(keep_forms)]
        if len(df):
            parts.append(pd.DataFrame({"ticker": tk, "date": df["filingDate"],
                                       "form": df["form"]}))
    if not parts:
        return pd.DataFrame(columns=["ticker", "date", "form"])
    return pd.concat(parts, ignore_index=True)


def build_label_events(suspensions: pd.DataFrame,
                       filings: pd.DataFrame | None = None,
                       delisting_reasons: dict[str, str] | None = None,
                       enforcement: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble the positive class.

    `delisting_reasons` maps ticker -> reason so benign Form 25 filings can be
    excluded. Without it, delistings are NOT promoted to positives, because an
    unfiltered Form 25 is roughly as likely to be a completed merger as an
    adverse termination.
    """
    frames = []
    if suspensions is not None and len(suspensions):
        s = suspensions.dropna(subset=["ticker"])[["ticker", "date"]].copy()
        s["kind"] = "sec_suspension"
        frames.append(s)
    if filings is not None and len(filings) and delisting_reasons:
        d = filings[filings["form"].isin(DELISTING_FORMS)][["ticker", "date"]].copy()
        adverse = {t for t, why in delisting_reasons.items()
                   if why not in ("merger", "going_private", "acquired", "voluntary")}
        d = d[d["ticker"].isin(adverse)]
        d["kind"] = "delisting"
        frames.append(d)
    if enforcement is not None and len(enforcement):
        e = enforcement[["ticker", "date"]].copy()
        e["kind"] = "enforcement"
        frames.append(e)
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "kind"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date", "ticker"]).drop_duplicates().reset_index(drop=True)
