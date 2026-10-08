"""
api.py - serves the engine's signals to downstream applications (Module A/B, dashboards).

Run:   uvicorn api:app --reload          then open http://127.0.0.1:8000/docs

Endpoints
    GET /health                 service + data status
    GET /signals                filter / sort / page through signals
    GET /signals/latest         newest live signals (only rows that have a timestamp)
    GET /signals/{id}           one signal
    GET /stats                  counts and mean impact by event type

Examples
    /signals?min_impact=7                        high-impact signals (stress-test trigger)
    /signals?event_type=Geopolitical&live_only=true&sort=time
    /signals/latest?limit=10
    /signals?event_type=mna                      "M&A" can also be written ma / mna

The file is re-read automatically when data/signals.jsonl changes, so re-running the
pipeline updates the API without a restart.
"""
import json
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

SIGNALS_PATH = Path(os.environ.get("SIGNALS_PATH", "data/signals.jsonl"))

# Fields returned to clients (the outline's record shape plus a few useful extras).
PUBLIC_FIELDS = ["id", "timestamp", "source", "headline", "entities", "tickers", "sentiment",
                 "direction", "event_type", "event_confidence", "impact_score", "impact_components"]
EVENT_ALIASES = {"mna": "ma", "mergersacquisitions": "ma"}


def _norm(s: str) -> str:
    """'M&A' -> 'ma', 'Credit Event' -> 'creditevent', 'credit_event' -> 'creditevent'."""
    s = re.sub(r"[^a-z]", "", s.lower())
    return EVENT_ALIASES.get(s, s)


def _project(row: dict) -> dict:
    out = {k: row.get(k) for k in PUBLIC_FIELDS}
    out["entities"] = row.get("entities") or []   # filled by preprocess.py if that stage exists
    out["tickers"] = row.get("tickers") or []
    out["live"] = row.get("source") != "kaggle_csv"
    return out


class SignalStore:
    """Loads signals.jsonl and reloads it whenever the file changes."""

    def __init__(self, path: Path):
        self.path, self._mtime, self._rows = path, None, []

    def rows(self) -> list:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._mtime, self._rows = None, []
            return self._rows
        if mtime != self._mtime:
            with open(self.path, encoding="utf-8") as f:
                self._rows = [_project(json.loads(line)) for line in f if line.strip()]
            self._mtime = mtime
        return self._rows


def query_signals(rows, event_type=None, min_impact=None, max_impact=None, direction=None,
                  source=None, ticker=None, live_only=False, sort="impact"):
    """Filter and sort. Returns the full matching list (the endpoint applies paging)."""
    out = rows
    if event_type:
        want = _norm(event_type)
        out = [r for r in out if _norm(r["event_type"] or "") == want]
    if min_impact is not None:
        out = [r for r in out if r["impact_score"] >= min_impact]
    if max_impact is not None:
        out = [r for r in out if r["impact_score"] <= max_impact]
    if direction:
        out = [r for r in out if r["direction"] == direction.lower()]
    if source:
        out = [r for r in out if source.lower() in (r["source"] or "").lower()]
    if ticker:
        out = [r for r in out if ticker.upper() in [t.upper() for t in r["tickers"]]]
    if live_only:
        out = [r for r in out if r["live"]]
    if sort == "time":
        out = sorted(out, key=lambda r: r["timestamp"] or "", reverse=True)
    else:
        out = sorted(out, key=lambda r: r["impact_score"], reverse=True)
    return out


def latest_signals(rows, limit=20):
    """Newest first. Rows without a timestamp (the Kaggle CSV) are not 'latest' by definition."""
    stamped = [r for r in rows if r["timestamp"]]
    return sorted(stamped, key=lambda r: r["timestamp"], reverse=True)[:limit]


def stats(rows, threshold=7.0):
    by_type = {}
    for r in rows:
        d = by_type.setdefault(r["event_type"], {"count": 0, "_sum": 0.0, "high_impact": 0})
        d["count"] += 1
        d["_sum"] += r["impact_score"]
        d["high_impact"] += r["impact_score"] > threshold
    for d in by_type.values():
        d["mean_impact"] = round(d.pop("_sum") / d["count"], 2)
    return {"total": len(rows), "live": sum(r["live"] for r in rows),
            "high_impact_threshold": threshold, "by_event_type": by_type}


# ---------------------------------------------------------------- FastAPI app
store = SignalStore(SIGNALS_PATH)
app = FastAPI(title="Risk Engine API", version="1.0",
              description="Structured financial risk signals from unstructured news.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@app.get("/health")
def health():
    rows = store.rows()
    return {"status": "ok" if rows else "no data - run `python pipeline.py` first",
            "signals": len(rows), "source_file": str(SIGNALS_PATH)}


@app.get("/signals")
def get_signals(
    event_type: Optional[str] = Query(None, description="Geopolitical, Macroeconomic, Credit Event, M&A, Product Launch, Other"),
    min_impact: Optional[float] = Query(None, ge=1, le=10),
    max_impact: Optional[float] = Query(None, ge=1, le=10),
    direction: Optional[str] = Query(None, description="negative | neutral | positive"),
    source: Optional[str] = Query(None, description="substring match, e.g. yahoo or gdelt"),
    ticker: Optional[str] = Query(None),
    live_only: bool = Query(False, description="exclude the Kaggle CSV rows"),
    sort: str = Query("impact", description="impact | time"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    matched = query_signals(store.rows(), event_type, min_impact, max_impact, direction,
                            source, ticker, live_only, sort)
    return {"count": len(matched), "limit": limit, "offset": offset,
            "signals": matched[offset:offset + limit]}


@app.get("/signals/latest")   # declared before /signals/{signal_id} so "latest" is not read as an id
def get_latest(limit: int = Query(20, ge=1, le=200)):
    sigs = latest_signals(store.rows(), limit)
    return {"count": len(sigs), "signals": sigs}


@app.get("/signals/{signal_id}")
def get_signal(signal_id: str):
    for r in store.rows():
        if r["id"] == signal_id:
            return r
    raise HTTPException(status_code=404, detail="signal not found")


@app.get("/stats")
def get_stats(threshold: float = Query(7.0, ge=1, le=10)):
    return stats(store.rows(), threshold)