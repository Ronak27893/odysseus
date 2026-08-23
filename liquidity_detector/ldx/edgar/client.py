"""EDGAR access: ticker->CIK map, per-company filing history, full-text search.

SEC access rules are not optional and are enforced here rather than left to
the caller: a declared User-Agent carrying real contact information, and a
request rate under 10/second. Responses are cached on disk because the filing
history for a multi-thousand-name universe is tens of thousands of requests
and must be re-runnable without re-fetching.

Not exercised against the live API in environments whose egress policy blocks
sec.gov; request shapes follow the published EDGAR REST interfaces.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"
FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SUSPENSIONS_PAGE = "https://www.sec.gov/litigation/suspensions"

#: Forms that matter to this model.
OFFERING_FORMS = ("424B5", "424B3", "424B4", "S-1", "S-3", "S-1/A", "S-3/A")
DELISTING_FORMS = ("25-NSE", "25")
EVENT_FORMS = ("8-K",)


class EdgarClient:
    def __init__(self, user_agent: str | None = None,
                 cache_dir: str | Path = ".cache/edgar",
                 rate_limit_per_sec: float = 8.0, max_retries: int = 4):
        ua = user_agent or os.environ.get("SEC_USER_AGENT")
        if not ua or "@" not in ua:
            raise RuntimeError(
                "SEC requires a User-Agent with contact information, e.g. "
                "'Research Group research@example.com'. Set SEC_USER_AGENT.")
        self.headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._min_interval = 1.0 / rate_limit_per_sec
        self._last = 0.0
        self.max_retries = max_retries
        self._session = requests.Session()

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _get(self, url: str, params: dict | None = None, cache_key: str | None = None):
        path = (self.cache / f"{cache_key}.json") if cache_key else None
        if path and path.exists():
            return json.loads(path.read_text())
        delay = 1.0
        for _ in range(self.max_retries):
            self._throttle()
            r = self._session.get(url, params=params, headers=self.headers, timeout=45)
            if r.status_code == 200:
                try:
                    payload = r.json()
                except ValueError:
                    payload = {"_text": r.text}
                if path:
                    path.write_text(json.dumps(payload))
                return payload
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay); delay *= 2; continue
            r.raise_for_status()
        raise RuntimeError(f"EDGAR request failed after {self.max_retries} attempts: {url}")

    # -- reference ---------------------------------------------------------
    def ticker_cik_map(self) -> pd.DataFrame:
        payload = self._get(COMPANY_TICKERS, cache_key="company_tickers")
        rows = payload.values() if isinstance(payload, dict) else payload
        df = pd.DataFrame(list(rows))
        return df.rename(columns={"cik_str": "cik", "ticker": "ticker", "title": "name"})

    def company_filings(self, cik: int) -> pd.DataFrame:
        """Full filing history for one CIK, including the older archive files."""
        payload = self._get(SUBMISSIONS.format(cik=int(cik)), cache_key=f"sub_{int(cik):010d}")
        recent = payload.get("filings", {}).get("recent", {})
        frames = [pd.DataFrame(recent)] if recent else []
        for extra in payload.get("filings", {}).get("files", []) or []:
            name = extra.get("name")
            if not name:
                continue
            more = self._get(f"https://data.sec.gov/submissions/{name}",
                             cache_key=f"sub_extra_{name.replace('/', '_')}")
            if more:
                frames.append(pd.DataFrame(more))
        if not frames:
            return pd.DataFrame(columns=["form", "filingDate", "accessionNumber"])
        df = pd.concat(frames, ignore_index=True)
        keep = [c for c in ("form", "filingDate", "accessionNumber", "primaryDocument")
                if c in df.columns]
        df = df[keep].copy()
        df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
        df["cik"] = int(cik)
        return df.dropna(subset=["filingDate"])

    def full_text_search(self, query: str, forms: str | None = None,
                         date_from: str | None = None, date_to: str | None = None,
                         limit: int = 100) -> pd.DataFrame:
        """EDGAR full-text search. Covers 2001 onward only."""
        params = {"q": query, "from": 0, "size": min(limit, 100)}
        if forms:
            params["forms"] = forms
        if date_from:
            params["dateRange"] = "custom"; params["startdt"] = date_from
        if date_to:
            params["enddt"] = date_to
        payload = self._get(FULL_TEXT_SEARCH, params=params)
        hits = (payload.get("hits", {}) or {}).get("hits", []) or []
        rows = []
        for h in hits:
            src = h.get("_source", {})
            rows.append({
                "accession": h.get("_id"),
                "form": src.get("root_form") or src.get("file_type"),
                "date": pd.to_datetime(src.get("file_date"), errors="coerce"),
                "cik": (src.get("ciks") or [None])[0],
                "display_names": "; ".join(src.get("display_names") or []),
            })
        return pd.DataFrame(rows)

    def trading_suspensions_html(self) -> str:
        """Raw HTML of the SEC trading-suspension list, for `parse_suspension_table`."""
        payload = self._get(SUSPENSIONS_PAGE, cache_key="suspensions_page")
        return payload.get("_text", "") if isinstance(payload, dict) else str(payload)
