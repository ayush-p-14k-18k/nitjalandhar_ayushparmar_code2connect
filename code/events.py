"""Stage 4: event classification.

Layer 1: keyword rules (cheap, explainable, word-boundary regex, weighted).
Layer 2: zero-shot (facebook/bart-large-mnli) only for headlines the rules miss.

Adds to each record: event_type, event_confidence, event_method, event_hits
Usage:  python events.py [--no-zeroshot] [--skip-source kaggle_csv]
"""
import argparse, json, re
from collections import Counter

CATEGORIES = ["Geopolitical", "Macroeconomic", "Credit Event", "M&A", "Product Launch"]
OTHER = "Other"  # earnings, analyst calls, generic corporate news

# (pattern, weight). Weight 2 = strong signal, 1 = weak/ambiguous.
RULES = {
    "Geopolitical": [
        (r"sanction\w*", 2), (r"\bwar\b", 2), (r"invasion|invade\w*", 2),
        (r"missile|airstrike|military|ceasefire|troops", 2),
        (r"tariff\w*", 2), (r"trade (war|dispute|tension\w*|deal)", 2),
        (r"export (ban|control\w*)", 2), (r"embargo", 2), (r"geopolitic\w*", 2),
        (r"\b(russia\w*|ukrain\w*|china|chinese|taiwan|iran\w*|israel\w*|palestin\w*|gaza|lebanon|hezbollah|houthi|hamas|opec|brics|nato|putin|netanyahu)\b", 1),
        (r"\b(nuclear|conflict|crackdown|tankers?|foreign minister|diplomat\w*)\b", 1),
        (r"terror\w*|hijack\w*", 2),
        (r"\battack(s|ed)?\b|\bstrikes? on\b|\bhostilit\w*", 1),
    ],
    "Macroeconomic": [
        (r"rate (hike|cut|decision)s?", 2), (r"interest rates?|prime rate", 2),
        (r"inflation|\bcpi\b|\bppi\b", 2), (r"recession", 2),
        (r"\bgdp\b", 2), (r"jobs report|unemployment|nonfarm|payrolls", 2),
        (r"\bfomc\b|federal reserve|\bfed\b|central bank|\becb\b", 2),
        (r"treasury yields?|bond yields?", 2),
        (r"bond (market|swings?|sell-?off|rout)|bond issuance|treasury borrowing", 2), (r"debt crisis|debt ceiling", 2),
        (r"\b(dollar|oil prices?|crude)\b", 1),
    ],
    "Credit Event": [
        (r"\bdefault\w*", 2), (r"bankrupt\w*|chapter 11|insolven\w*", 2),
        (r"credit (rating|downgrade)|rating downgrade", 2),
        (r"(moody|fitch|s&p)\W*s?\b.{0,40}(downgrad|cut|lower)", 2),
        (r"downgrad\w* (debt|bonds?|its credit)", 2),
        (r"debt restructur\w*|covenant|write-?offs?|loan loss|bad loans?", 2),
        (r"\bCLO\b|junk bond|credit spreads?", 1),
        (r"\b(prices?|issues?|raises?)\b.{0,25}\b(loan|notes|bonds|debt)\b", 2),
    ],
    "M&A": [
        (r"\bacqui(re|res|red|sition|sitions)\b", 2), (r"merger|\bmerge\w*", 2),
        (r"takeover|buyout|take-private", 2), (r"\bspac\b", 2),
        (r"\bbid(s|ding)?\b", 1), (r"agrees? to (sale|sell|buy)", 2),
        (r"\bto (buy|sell)\b", 1), (r"\bdivest\w*|spin-?off", 2), (r"\bsale\b", 1),
    ],
    "Product Launch": [
        (r"launch\w*", 2), (r"unveil\w*", 2), (r"introduc\w+ (new|its)", 2),
        (r"\brolls? out\b|\brollout\b", 2), (r"new (chips?|product|model|version)", 1),
        (r"wins? (fda|eu) approval|approvals?\b", 1),
    ],
}
# Routine corporate news. Overrides only weak M&A / Product Launch matches, so a
# genuine credit, macro or geopolitical signal is never hidden by it.
OTHER_RX = re.compile(
    r"downgrad\w*|upgrad\w*|price target|\banalysts?\b|\bsees \d+% upside"
    r"|\b(signs?|inks?|wins?|extends?|renews?)\b.{0,60}\b(deal|contract|agreement)\b"
    r"|\bbuyback\b|\bdividend\b|\bearnings\b|\bQ[1-4]\b"
    r"|sequel|crossover|spoilers|\bmipcom\b|game of thrones|sweepstakes", re.I)
COMPILED = {c: [(re.compile(p, re.I), w) for p, w in rs] for c, rs in RULES.items()}
# Tie-break order when scores are equal: more systemic events win.
PRIORITY = ["Credit Event", "Geopolitical", "Macroeconomic", "M&A", "Product Launch"]

# Descriptive labels help NLI zero-shot much more than bare category names.
ZS_LABELS = {
    "Geopolitical": "geopolitics, war, sanctions or trade policy",
    "Macroeconomic": "macroeconomics, interest rates, inflation or central banks",
    "Credit Event": "credit risk, debt default, bankruptcy or a credit rating change",
    "M&A": "a merger, acquisition or company sale",
    "Product Launch": "a new product launch or product announcement",
    OTHER: "everyday news unrelated to markets, such as personal finance, entertainment, crime, sports, or routine company contracts, earnings and analyst opinion",
}
# Zero-shot only counts if confident. Costly false alarms (Credit, Geopolitical, Macro
# feed the stress test) need more confidence. Heuristic values, not tuned on labelled data.
ZS_THRESHOLDS = {"default": 0.60, "Credit Event": 0.95, "Geopolitical": 0.70, "Macroeconomic": 0.70}
ZS_CACHE_PATH = "data/zs_cache.json"
_zs = None


def rule_classify(text):
    scores, hits = {}, {}
    for cat, pats in COMPILED.items():
        for rx, w in pats:
            m = rx.search(text)
            if m:
                scores[cat] = scores.get(cat, 0) + w
                hits.setdefault(cat, []).append(m.group(0).lower())
    # Weak-only matches (total weight 1) are too ambiguous to trust alone.
    scores = {c: s for c, s in scores.items() if s >= 2}
    om = OTHER_RX.search(text)
    if om and (not scores or max(scores, key=scores.get) in ("M&A", "Product Launch")):
        return OTHER, 0.70, [om.group(0).lower()]
    if not scores:
        return None
    best = max(scores, key=lambda c: (scores[c], -PRIORITY.index(c)))
    top = scores[best]
    runner_up = max([s for c, s in scores.items() if c != best], default=0)
    # Confidence grows with evidence, shrinks when a competing category is close.
    conf = min(0.95, 0.60 + 0.08 * top - 0.05 * runner_up)
    return best, round(conf, 2), hits[best]


_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        try:
            _cache = json.load(open(ZS_CACHE_PATH, encoding="utf-8"))
        except (OSError, ValueError):
            _cache = {}
    return _cache


def save_cache():
    if _cache is not None:
        json.dump(_cache, open(ZS_CACHE_PATH, "w", encoding="utf-8"), ensure_ascii=False)


def zero_shot(text):
    """Raw model output is cached, so thresholds can be tuned without re-running BART."""
    global _zs
    cache = _load_cache()
    if text not in cache:
        if _zs is None:
            from transformers import pipeline
            _zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        out = _zs(text, list(ZS_LABELS.values()), hypothesis_template="This news is about {}.")
        inv = {v: k for k, v in ZS_LABELS.items()}
        cache[text] = [inv[out["labels"][0]], float(out["scores"][0])]
    cat, score = cache[text]
    if score < ZS_THRESHOLDS.get(cat, ZS_THRESHOLDS["default"]):
        cat = OTHER
    return cat, round(score, 2)


def classify(text, use_zeroshot=True):
    r = rule_classify(text)
    if r:
        return {"event_type": r[0], "event_confidence": r[1],
                "event_method": "rules", "event_hits": r[2]}
    if use_zeroshot:
        cat, conf = zero_shot(text)
        return {"event_type": cat, "event_confidence": conf,
                "event_method": "zero_shot", "event_hits": []}
    return {"event_type": OTHER, "event_confidence": 0.3,
            "event_method": "none", "event_hits": []}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/scored.jsonl")
    ap.add_argument("--out", default="data/classified.jsonl")
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--skip-source", default="kaggle_csv",
                    help="source to run rules-only on (keeps slow model off the CSV)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    todo = sum(1 for r in rows if r["source"] != a.skip_source)
    print(f"{len(rows)} rows ({todo} eligible for zero-shot)")
    done = 0
    for r in rows:
        zs_ok = not a.no_zeroshot and r["source"] != a.skip_source
        r.update(classify(r["headline"], zs_ok))
        if r["event_method"] == "zero_shot":
            done += 1
            if done % 10 == 0:
                print(f"  zero-shot {done} headlines done...", flush=True)
    save_cache()
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(Counter(r["event_type"] for r in rows))
    print(Counter(r["event_method"] for r in rows))
    live = [x for x in rows if x["source"] != a.skip_source]
    print("\nlive:", Counter(r["event_type"] for r in live))
    live.sort(key=lambda r: r["event_type"] == OTHER)  # real events first
    for r in live[:40]:
        print(f'{r["event_type"]:<14}{r["event_confidence"]:.2f} {r["event_method"]:<9}{r["headline"][:70]}')