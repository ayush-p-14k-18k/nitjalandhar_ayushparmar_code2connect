"""
impact.py - Stage 5 of the AI/NLP risk engine.

Reads data/classified.jsonl (sentiment + event labels), writes data/signals.jsonl.

There is NO labelled dataset for "market severity", so the score is a transparent,
hand-set formula (documented below), not a trained model. Every component is stored in
`impact_components` so any score can be explained.

    prior    = base_severity(event_type) * (0.7 + 0.3 * event_confidence)
    evidence = 0.5 * |sentiment|
             + 0.3 * min(2, strong_words) / 2      (war, default, collapse, ...)
             + 0.2 * min(2, other_sources) / 2     (same story in several outlets)
    raw      = prior * (1 + evidence)               evidence matters more for serious event types
    impact   = 1 + 9 * (1 - exp(-raw / 7))          soft saturation: ordering kept near the top,
                                                    unlike a hard clip at 10
    -> range 1..10. `direction` (negative/positive/neutral) comes from the sentiment sign.

Usage:  python impact.py [--inp data/classified.jsonl] [--out data/signals.jsonl]
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict

BASE_SEVERITY = {
    "Credit Event": 7.0,
    "Geopolitical": 6.0,
    "Macroeconomic": 5.5,
    "M&A": 5.0,
    "Product Launch": 3.0,
    "Other": 1.5,
}
W_SENT, W_INTENSITY, W_COVERAGE = 0.5, 0.3, 0.2
SATURATION_K = 7.0
INTENSITY_RX = re.compile(
    r"\b(collaps\w*|default\w*|bankrupt\w*|insolven\w*|invasion|invade\w*|war|attack\w*|"
    r"crash\w*|plung\w*|plummet\w*|crisis|sanction\w*|missile\w*|terror\w*|hijack\w*|"
    r"recession|meltdown|sell-?off|shock\w*|ban)\b", re.I)
STOP = set("the a an of to in on for and or with as at by from is are its it be this that "
           "why how can could will what who after over amid new says say".split())


def tokens(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 and w not in STOP}


def coverage_counts(rows, skip_source="kaggle_csv", min_jaccard=0.4):
    """For each live row: number of OTHER sources carrying a similar headline."""
    live = [(i, r["source"], tokens(r["headline"])) for i, r in enumerate(rows)
            if r["source"] != skip_source]
    seen = defaultdict(set)
    for a in range(len(live)):
        for b in range(a + 1, len(live)):
            (i, si, ti), (j, sj, tj) = live[a], live[b]
            if si == sj or not ti or not tj:
                continue
            if len(ti & tj) / len(ti | tj) >= min_jaccard:
                seen[i].add(sj)
                seen[j].add(si)
    return {i: len(s) for i, s in seen.items()}


def impact_score(row, n_other_sources=0):
    base = BASE_SEVERITY.get(row["event_type"], BASE_SEVERITY["Other"])
    prior = base * (0.7 + 0.3 * row.get("event_confidence", 0.5))
    n_words = min(2, len(INTENSITY_RX.findall(row["headline"])))
    n_src = min(2, n_other_sources)
    evidence = (W_SENT * abs(row["sentiment"]) + W_INTENSITY * n_words / 2
                + W_COVERAGE * n_src / 2)
    raw = prior * (1 + evidence)
    score = round(1 + 9 * (1 - math.exp(-raw / SATURATION_K)), 1)
    return score, {"prior": round(prior, 2), "evidence": round(evidence, 2),
                   "strong_words": n_words, "other_sources": n_src, "raw": round(raw, 2)}


def direction(sentiment):
    return "negative" if sentiment <= -0.15 else "positive" if sentiment >= 0.15 else "neutral"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/classified.jsonl")
    ap.add_argument("--out", default="data/signals.jsonl")
    ap.add_argument("--show", type=int, default=25, help="live rows to print for calibration")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    cov = coverage_counts(rows)
    for i, r in enumerate(rows):
        r["impact_score"], r["impact_components"] = impact_score(r, cov.get(i, 0))
        r["direction"] = direction(r["sentiment"])

    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} signals -> {a.out}")

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["event_type"]].append(r["impact_score"])
    print("\nmean impact by event type:")
    for t, v in sorted(by_type.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        print(f"  {t:<15}{sum(v) / len(v):4.1f}  (n={len(v)}, max={max(v)})")
    print("\nhigh impact (>7):", Counter(r["event_type"] for r in rows if r["impact_score"] > 7))

    live = sorted((r for r in rows if r["source"] != "kaggle_csv"),
                  key=lambda r: -r["impact_score"])
    print(f"\ntop {a.show} live headlines by impact:")
    for r in live[:a.show]:
        print(f'{r["impact_score"]:>4}  {r["event_type"]:<14}{r["sentiment"]:+.2f}  {r["headline"][:70]}')