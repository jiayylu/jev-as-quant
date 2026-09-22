"""SEC EDGAR 8-K press releases: official, time-stamped company news (public domain).

Access follows the SEC fair-access policy: every request declares who is asking through a
User-Agent of the form "<name> <email>" (set SEC_USER_AGENT; requests without it are refused
by EDGAR), and we stay under 10 requests per second. Raw responses are cached on disk.

Timing: `acceptanceDateTime` in the submissions API is true UTC (Apple's 16:30 US/Eastern
earnings releases show as 20:30Z in summer and 21:30Z in winter), so it is converted to
US/Eastern before deciding which market open is the first tradable one.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

ET = "America/New_York"


class SecClient:
    def __init__(self, user_agent: str | None = None, cache_dir: str | Path = "data_cache/sec",
                 max_per_second: float = 9.0):
        self.ua = user_agent or os.environ.get("SEC_USER_AGENT", "")
        if "@" not in self.ua:
            raise RuntimeError('set SEC_USER_AGENT to "<your name or org> <your email>" (SEC fair-access policy)')
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_gap = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._last = 0.0
        self._local = threading.local()

    def _wait(self):
        with self._lock:
            now = time.monotonic()
            sleep = self._last + self.min_gap - now
            if sleep > 0:
                time.sleep(sleep)
            self._last = time.monotonic()

    def _conn(self, host: str):
        """One persistent HTTPS connection per thread and host (no TLS handshake per request)."""
        import http.client

        conns = getattr(self._local, "conns", None)
        if conns is None:
            conns = self._local.conns = {}
        if host not in conns:
            conns[host] = http.client.HTTPSConnection(host, timeout=30)
        return conns[host]

    def get(self, url: str, retries: int = 7) -> bytes | None:
        """Cached GET. Returns None for 404 (e.g. a filing without the expected document)."""
        import http.client
        from urllib.parse import urlsplit

        key = self.cache / (hashlib.sha1(url.encode()).hexdigest() + ".gz")
        if key.exists():
            data = gzip.decompress(key.read_bytes())
            return None if data == b"__404__" else data
        u = urlsplit(url)
        path = u.path + (f"?{u.query}" if u.query else "")
        delay = 1.0
        for attempt in range(retries):
            self._wait()
            try:
                c = self._conn(u.netloc)
                c.request("GET", path, headers={"User-Agent": self.ua, "Accept-Encoding": "gzip",
                                                "Host": u.netloc, "Connection": "keep-alive"})
                r = c.getresponse()
                data = r.read()
                if r.status == 200:
                    if r.getheader("Content-Encoding") == "gzip":
                        data = gzip.decompress(data)
                    key.write_bytes(gzip.compress(data))
                    return data
                if r.status == 404:
                    key.write_bytes(gzip.compress(b"__404__"))
                    return None
                if r.status not in (403, 429, 500, 502, 503) or attempt == retries - 1:
                    raise urllib.error.HTTPError(url, r.status, r.reason, r.headers, None)
            except (http.client.HTTPException, OSError):
                self._local.conns.pop(u.netloc, None)  # drop the broken connection and reconnect
                if attempt == retries - 1:
                    raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        return None

    def json(self, url: str):
        data = self.get(url)
        return None if data is None else json.loads(data)


# ------------------------------------------------------------------------------ universe


def sp500_from_wikitext(wikitext: str) -> pd.DataFrame:
    """Parse Wikipedia's 'List of S&P 500 companies' (action=raw): symbol, name, sector, added, cik.

    Cells are found by content (the CIK is the only all-digit cell, the join date the only ISO
    date) because a few rows carry extra or missing cells.
    """
    table = wikitext.split('id="constituents"', 1)[1].split("|}", 1)[0]
    rows = []
    for block in table.split("\n|-")[1:]:
        cells = [c.strip() for c in re.split(r"\n\|\|?", "\n" + block.strip()) if c.strip()]
        if not cells or "Symbol|" not in cells[0]:
            continue
        sym = re.search(r"Symbol\|([^}|]+)", cells[0]).group(1).strip()
        name = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", cells[1]).strip() if len(cells) > 1 else ""
        cik = next((c for c in cells[2:] if re.fullmatch(r"\d{5,10}", c)), None)
        added = next((c[:10] for c in cells[2:] if re.match(r"\d{4}-\d{2}-\d{2}", c)), "")
        if cik is None:
            continue
        rows.append({"symbol": sym, "name": name, "sector": cells[2] if len(cells) > 2 else "",
                     "added": added, "cik": cik.zfill(10)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ filings


def list_8k(client: SecClient, cik: str, since: str) -> pd.DataFrame:
    """All original 8-Ks accepted on/after `since` (UTC date), with item codes and timestamps."""
    base = f"https://data.sec.gov/submissions/CIK{cik}.json"
    sub = client.json(base)
    if sub is None:
        return pd.DataFrame()
    pages = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        if f.get("filingTo", "9999") >= since:
            extra = client.json(f"https://data.sec.gov/submissions/{f['name']}")
            if extra:
                pages.append(extra)
    frames = []
    for p in pages:
        df = pd.DataFrame({k: p[k] for k in ("accessionNumber", "filingDate", "acceptanceDateTime", "form",
                                             "items", "primaryDocument")})
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("accessionNumber")
    df = df[(df["form"] == "8-K") & (df["acceptanceDateTime"].str[:10] >= since)].copy()
    df["accepted_utc"] = pd.to_datetime(df["acceptanceDateTime"], utc=True)
    df["accepted_et"] = df["accepted_utc"].dt.tz_convert(ET)
    df["cik"] = cik
    return df.reset_index(drop=True)


class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "td", "title"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    text = "".join(p.parts).replace("\xa0", " ")
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))


BOILERPLATE = re.compile(
    r"^(exhibit\s*\d|ex-?\s?99|press release|news release|for immediate release|investor (contact|relations)"
    r"|media (contact|relations)|contacts?\b|page \d|table of contents|source:|\(?(nyse|nasdaq)\b|www\.|https?://"
    r"|forward[- ]looking statements|refer to:|c/o\b|on receipt\b)", re.I)
_DATELINE = re.compile(r"^(january|february|march|april|may|june|july|august|september|october|november|december)"
                       r"\s+\d{1,2},?\s+\d{4}\b", re.I)
_CONTACT = re.compile(r"@|\(\d{3}\)\s*\d{3}-\d{4}|\b\d{3}[-.]\d{3}[-.]\d{4}\b")
_FILENAME = re.compile(r"^\S+\.(htm|html|txt|pdf|jpg|png)$", re.I)
_ADDRESS = re.compile(r"\b\d{5}(-\d{4})?$|\b(suite|floor|street|avenue|boulevard|p\.o\. box)\b", re.I)
_COMPANY_ONLY = re.compile(r"^[\w\s&.,'’-]{3,60}\b(inc|corp|corporation|company|co|plc|ltd|llc|n\.v|s\.a|holdings)\.?$", re.I)
_PERSON_TITLE = re.compile(r",\s*(vp|svp|evp|vice president|treasurer|cfo|ceo|chief|director|president|head of)\b", re.I)


def _is_headline(line: str) -> bool:
    n = len(line.split())
    if not 4 <= n <= 25:
        return False
    if _FILENAME.match(line) or _ADDRESS.search(line) or _COMPANY_ONLY.match(line):
        return False
    if (_PERSON_TITLE.search(line) or _DATELINE.match(line)) and n <= 8:
        return False
    return True


def lead_of_release(text: str, max_words: int = 110) -> tuple[str, str]:
    """(headline, lead) of a press release: first headline-like line, then the opening paragraph(s).

    Filters what EDGAR exhibits put before the headline: exhibit labels, file names, logos'
    alt text, addresses, datelines, company names, contact lines. Returns ("", "") when no line
    looks like a headline (e.g. slide decks, foreign-exchange filing forms).
    """
    lines = [l for l in text.split("\n") if len(l) >= 25 and not BOILERPLATE.match(l) and not _CONTACT.search(l)
             and not _FILENAME.match(l)]
    head = next((l for l in lines if _is_headline(l)), None)
    if head is None:
        return "", ""
    rest = lines[lines.index(head) + 1:]
    words: list[str] = []
    for l in rest:
        if len(l.split()) < 8:  # sub-headings, bullet fragments, datelines alone
            continue
        words += l.split()
        if len(words) >= max_words:
            break
    return head.strip(), " ".join(words[:max_words])


_ROW = re.compile(r'<td[^>]*>\s*<a href="([^"]+)">([^<]+)</a>[^<]*</td>\s*<td[^>]*>([^<]*)</td>', re.I)


def exhibit99(client: SecClient, cik: str, accession: str) -> dict | None:
    """Headline + lead paragraph of the EX-99.x press release attached to an 8-K, if any.

    The filing's index page is the only reliable map from file to exhibit type (file names
    follow no convention, e.g. NVIDIA's earnings release is `q2fy27pr.htm`).
    """
    cik_i = str(int(cik))
    acc = accession.replace("-", "")
    page = client.get(f"https://www.sec.gov/Archives/edgar/data/{cik_i}/{acc}/{accession}-index.htm")
    if page is None:
        return None
    docs = [(href, name.strip(), typ.strip().upper()) for href, name, typ in _ROW.findall(page.decode("utf-8", "ignore"))]
    ex = [d for d in docs if d[2].startswith("EX-99") and d[1].lower().endswith((".htm", ".html", ".txt"))]
    if not ex:
        return None
    ex.sort(key=lambda d: (d[2] != "EX-99.1", d[2]))
    href, name, typ = ex[0]
    url = href if href.startswith("http") else "https://www.sec.gov" + href
    raw = client.get(url)
    if raw is None:
        return None
    head, lead = lead_of_release(html_to_text(raw.decode("utf-8", errors="ignore")))
    if not head:
        return None
    return {"document": name, "exhibit": typ, "headline": head, "lead": lead}


def first_tradable_open(accepted_et: pd.Series, trading_days: pd.DatetimeIndex, conservative: bool = False) -> pd.Series:
    """First market open the news could be traded at, from US/Eastern acceptance times.

    Default: before 09:30 on a trading day -> that day's open; otherwise the next trading day.
    conservative=True: always the first trading day strictly after the acceptance date.
    """
    import numpy as np

    d = accepted_et.dt.tz_localize(None).dt.normalize()
    before_open = (accepted_et.dt.hour * 60 + accepted_et.dt.minute) < 9 * 60 + 30
    days = trading_days.values
    same = np.searchsorted(days, d.values, side="left")
    after = np.searchsorted(days, d.values, side="right")
    is_td = (same < len(days)) & (days[np.minimum(same, len(days) - 1)] == d.values)
    idx = after.copy()
    if not conservative:
        idx = np.where(is_td & before_open.values, same, after)
    out = pd.Series(pd.NaT, index=accepted_et.index, dtype="datetime64[ns]")
    ok = idx < len(days)
    out[ok] = days[idx[ok]]
    return out
