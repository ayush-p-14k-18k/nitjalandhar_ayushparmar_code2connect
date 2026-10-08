"""signals_client.py - subscribes to the risk engine (API first, local signals.jsonl as fallback)
and decides which signals trigger a stress test."""
import json
from pathlib import Path
import pandas as pd

KEEP = ["id", "timestamp", "source", "headline", "event_type", "event_confidence",
        "impact_score", "sentiment", "direction"]


def fetch_signals(api_url="http://127.0.0.1:8000", local_path="data/signals.jsonl", min_impact=1.0):
    """Returns (DataFrame, origin string)."""
    try:
        import requests
        r = requests.get(f"{api_url}/signals", params=dict(min_impact=max(min_impact, 1), limit=1000), timeout=4)
        r.raise_for_status()
        rows, origin = r.json()["signals"], f"API {api_url}"
    except Exception:
        p = Path(local_path)
        if not p.exists():
            return pd.DataFrame(columns=KEEP), "no data"
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        origin = f"file {local_path} (API unreachable)"
    df = pd.DataFrame(rows)
    for c in KEEP:
        if c not in df:
            df[c] = None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["live"] = df["source"].ne("kaggle_csv")
    return df[KEEP + ["live"]], origin


def triggers(df, threshold=6.5, event_types=None, live_only=True):
    """High-impact signals that should trigger a stress test, strongest first."""
    out = df[df["impact_score"] >= threshold]
    if event_types:
        out = out[out["event_type"].isin(event_types)]
    if live_only:
        out = out[out["live"]]
    return out.sort_values("impact_score", ascending=False)


def to_scenarios(trig):
    """One scenario per event type, using the strongest signal of that type (avoids counting
    twenty GDELT copies of the same Iran story twenty times)."""
    if trig.empty:
        return []
    top = trig.groupby("event_type")["impact_score"].max()
    return [(k, float(v)) for k, v in top.items()]
