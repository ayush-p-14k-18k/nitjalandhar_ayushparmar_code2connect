"""
ingest.py - Stage 1 of the AI/NLP risk engine.

Every source (CSV, RSS, NewsAPI, GDELT) is converted into one `Article` schema,
deduplicated, and written to data/articles.jsonl for the next stage.

The output file is MERGED with what is already there, so a failed live source
(rate limit, blocked feed) never erases earlier results.

Usage:
    python ingest.py                               # merge new live headlines into the file
    python ingest.py --recover data_backup/scored.jsonl   # re-add live rows lost earlier
    python ingest.py --no-gdelt                    # skip GDELT (e.g. while rate limited)
    python ingest.py --fresh                       # ignore the existing file (rebuild)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------- schema
@dataclass
class Article:
    id: str                          # unique per record (source + url/headline + time)
    source: str                      # e.g. "kaggle_csv", "yahoo_finance", "gdelt"
    headline: str
    body: str = ""
    timestamp: Optional[str] = None  # PUBLISH time, ISO 8601 UTC; None if dataset has no dates
    ingested_at: str = ""            # when WE saw it (always set)
    url: str = ""
    label: Optional[str] = None      # ground-truth sentiment if the dataset provides one
    content_hash: str = ""           # hash of normalized headline, used for dedup
    coverage: int = 1                # number of distinct sources reporting this story


ARTICLE_FIELDS = set(Article.__dataclass_fields__)


# ---------------------------------------------------------------- helpers
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_text(text) -> str:
    """Strip HTML tags and collapse whitespace. Keeps case, punctuation, negations."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def to_iso_utc(value) -> Optional[str]:
    """Parse almost any date format to 'YYYY-MM-DDTHH:MM:SSZ'. Returns None if unparseable."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def normalize_for_hash(headline: str) -> str:
    """Lowercase, drop punctuation: so the same story from different outlets matches."""
    return re.sub(r"[^a-z0-9 ]", "", headline.lower()).strip()


def make_article(source, headline, body="", timestamp=None, url="", label=None) -> Optional[Article]:
    headline = clean_text(headline)
    if not headline:
        return None
    body = clean_text(body)
    ts = to_iso_utc(timestamp)
    key = url or headline
    return Article(
        id=_md5(f"{source}|{key}|{ts}"),
        source=source,
        headline=headline,
        body=body,
        timestamp=ts,
        ingested_at=_now(),
        url=url,
        label=label,
        content_hash=_md5(normalize_for_hash(headline)),
    )


# ---------------------------------------------------------------- 1. CSV (Kaggle)
_CANDIDATES = {
    "headline": ["headline", "title", "text", "sentence", "news", "news headline"],
    "body": ["body", "content", "description", "summary", "article"],
    "time": ["timestamp", "date", "datetime", "published", "publishedat", "time"],
    "label": ["sentiment", "label", "class"],
}


def _pick(df: pd.DataFrame, wanted: Optional[str], key: str) -> Optional[str]:
    if wanted:
        return wanted
    lower = {c.lower(): c for c in df.columns}
    for cand in _CANDIDATES[key]:
        if cand in lower:
            return lower[cand]
    return None


def ingest_csv(path: str, source: str = "kaggle_csv", headline_col=None, body_col=None,
               time_col=None, label_col=None, names=None, encoding="utf-8") -> list[Article]:
    """
    Load a CSV with auto-detected columns (override via arguments).

    Headerless Kaggle file 'Sentiment Analysis for Financial News' (all-data.csv):
        ingest_csv("all-data.csv", names=["label", "headline"], encoding="latin-1")
    """
    df = pd.read_csv(path, names=names, header=None if names else "infer", encoding=encoding)
    h = _pick(df, headline_col, "headline")
    if h is None:
        raise ValueError(f"Could not find a headline column in {list(df.columns)}; pass headline_col=")
    b, t, l = _pick(df, body_col, "body"), _pick(df, time_col, "time"), _pick(df, label_col, "label")

    out = []
    for row in df.to_dict("records"):
        a = make_article(source, row[h], row.get(b, "") if b else "",
                         row.get(t) if t else None, label=str(row[l]).lower() if l else None)
        if a:
            out.append(a)
    return out


# ---------------------------------------------------------------- 2. RSS (live)
RSS_FEEDS = {
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    # Standard CNBC "Top News" feed. The older search.cnbc.com URL returned no entries.
    # (Not tested from here: if it still fails, the printed exception tells you why.)
    "cnbc_top_news": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    # Reuters' public RSS feeds were discontinued years ago, so they are not included.
}


def ingest_rss(feed_url: str, source_name: str) -> list[Article]:
    import feedparser
    feed = feedparser.parse(feed_url, agent="Mozilla/5.0 risk-engine-hackathon")
    if not feed.entries:
        print(f"[rss] {source_name}: no entries, skipping "
              f"(http status={feed.get('status', '?')}, error={getattr(feed, 'bozo_exception', None)!r})")
        return []
    out = []
    for e in feed.entries:
        published = None
        if getattr(e, "published_parsed", None):
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        a = make_article(source_name, e.get("title", ""), e.get("summary", e.get("description", "")),
                         published, url=e.get("link", ""))
        if a:
            out.append(a)
    return out


# ---------------------------------------------------------------- 3. NewsAPI (live)
def ingest_newsapi(query: str = "stocks OR economy OR sanctions", page_size: int = 100) -> list[Article]:
    """Needs env var NEWSAPI_KEY. Free tier has a daily request cap and delayed articles."""
    import requests
    key = os.environ.get("NEWSAPI_KEY")
    if not key:
        raise RuntimeError("Set NEWSAPI_KEY environment variable first.")
    r = requests.get("https://newsapi.org/v2/everything",
                     params={"q": query, "language": "en", "sortBy": "publishedAt",
                             "pageSize": page_size, "apiKey": key}, timeout=30)
    r.raise_for_status()
    out = []
    for item in r.json().get("articles", []):
        src = "newsapi:" + (item.get("source") or {}).get("name", "unknown")
        a = make_article(src, item.get("title"), item.get("description") or item.get("content"),
                         item.get("publishedAt"), url=item.get("url", ""))
        if a:
            out.append(a)
    return out


# ---------------------------------------------------------------- 4. GDELT (live)
# One query per event family, so live data covers more than geopolitics.
GDELT_QUERIES = {
    "geopolitical": '(sanctions OR war OR missile OR ceasefire OR tariffs) sourcelang:english',
    "macro": '("central bank" OR inflation OR "interest rates" OR recession) sourcelang:english',
    "credit": '(default OR bankruptcy OR "credit rating" OR downgrade) sourcelang:english',
}
GDELT_PAUSE_S = 6            # GDELT asks for at most one request per ~5 seconds
GDELT_RETRY_WAITS = (10, 20, 40)


def _gdelt_fetch(query: str, timespan: str, max_records: int):
    """Returns the list of article dicts, or None if rate limited / unusable."""
    import requests
    params = {"query": query, "mode": "artlist", "format": "json",
              "maxrecords": max_records, "timespan": timespan, "sort": "datedesc"}
    for attempt, wait in enumerate((0,) + GDELT_RETRY_WAITS):
        if wait:
            print(f"[gdelt] rate limited, waiting {wait}s (retry {attempt}/{len(GDELT_RETRY_WAITS)})")
            time.sleep(wait)
        r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params,
                         headers={"User-Agent": "risk-engine-hackathon"}, timeout=30)
        if r.status_code == 429:
            continue
        r.raise_for_status()
        try:
            return r.json().get("articles", [])
        except ValueError:   # GDELT answers some errors with plain text, not JSON
            print(f"[gdelt] non-JSON reply: {r.text[:100]!r}")
            return []
    return None


def ingest_gdelt(queries: dict = None, timespan: str = "1d", max_records: int = 100) -> list[Article]:
    """GDELT DOC 2.0 API: no key needed. Titles only. Keeps whatever it got if limited midway."""
    queries = queries or GDELT_QUERIES
    out = []
    for i, (name, q) in enumerate(queries.items()):
        if i:
            time.sleep(GDELT_PAUSE_S)
        items = _gdelt_fetch(q, timespan, max_records)
        if items is None:
            print(f"[gdelt] still rate limited on '{name}', stopping GDELT for this run")
            break
        print(f"[gdelt] {name}: {len(items)} articles")
        for item in items:
            seen = item.get("seendate", "")
            ts = datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc) if seen else None
            a = make_article("gdelt:" + item.get("domain", "unknown"), item.get("title"),
                             timestamp=ts, url=item.get("url", ""))
            if a:
                out.append(a)
    return out


# ---------------------------------------------------------------- load / dedup / output
def load_articles(path: str, live_only: bool = False) -> list[Article]:
    """Read articles from a JSONL file. Extra keys (sentiment, events...) are ignored,
    so scored.jsonl / classified.jsonl can be used to recover rows."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if live_only and d.get("source") == "kaggle_csv":
                continue
            out.append(Article(**{k: v for k, v in d.items() if k in ARTICLE_FIELDS}))
    return out


def deduplicate(articles: list[Article]) -> list[Article]:
    """
    Merge repeats of the same story (same normalized headline).
    Keeps the first copy (so older records win) and sets `coverage` = number of distinct sources.
    """
    kept: dict[str, Article] = {}
    sources: dict[str, set] = {}
    for a in articles:
        if a.content_hash not in kept:
            kept[a.content_hash] = a
            sources[a.content_hash] = set()
        sources[a.content_hash].add(a.source)
    for h, a in kept.items():
        a.coverage = max(a.coverage, len(sources[h]))   # never lower a coverage seen earlier
    return list(kept.values())


def write_jsonl(articles: list[Article], path: str = "data/articles.jsonl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
    print(f"wrote {len(articles)} articles -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/articles.jsonl")
    ap.add_argument("--fresh", action="store_true", help="ignore the existing output file")
    ap.add_argument("--recover", default=None,
                    help="re-add non-Kaggle rows from this JSONL (e.g. data_backup/scored.jsonl)")
    ap.add_argument("--no-gdelt", action="store_true")
    args = ap.parse_args()

    articles: list[Article] = [] if args.fresh else load_articles(args.out)
    print(f"starting from {len(articles)} existing articles")

    if args.recover:
        rec = load_articles(args.recover, live_only=True)
        print(f"recovered {len(rec)} live rows from {args.recover}")
        articles += rec

    if not any(a.source == "kaggle_csv" for a in articles):   # static data only once
        articles += ingest_csv("data/all-data.csv", names=["label", "headline"], encoding="latin-1")

    for name, url in RSS_FEEDS.items():
        try:
            articles += ingest_rss(url, name)
        except Exception as e:
            print(f"[rss] {name} failed: {e}")

    if not args.no_gdelt:
        try:
            articles += ingest_gdelt()
        except Exception as e:
            print(f"[gdelt] failed: {e}")

    write_jsonl(deduplicate(articles), args.out)