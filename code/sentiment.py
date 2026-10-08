"""
sentiment.py - Stage 3 of the AI/NLP risk engine.

Reads data/articles.jsonl, scores every headline with FinBERT, writes data/scored.jsonl.

Adds to each record:
    sentiment             float in [-1, +1]   = P(positive) - P(negative)
    sentiment_label       "positive" | "negative" | "neutral"
    sentiment_confidence  float in [0, 1]     = probability of the winning label
    sentiment_text        the exact text that was scored

By default, trailing clickbait questions are dropped before scoring
("Signs $700M Deal. Can Profits Grow?" -> "Signs $700M Deal."), because FinBERT
was trained on declarative sentences. Use --no-strip-questions to disable.

Usage:
    python sentiment.py --limit 200      # quick test on a random sample
    python sentiment.py                  # full run
    python sentiment.py --no-strip-questions
"""
import argparse
import json
import random
import re

import torch
from transformers import pipeline

MODEL = "ProsusAI/finbert"


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} records -> {path}")


def scoring_text(headline: str, strip_questions: bool = True) -> str:
    """Drop question sentences if at least one declarative sentence remains."""
    if not strip_questions:
        return headline
    parts = re.split(r"(?<=[.!?])\s+", headline.strip())
    kept = [p for p in parts if not p.endswith("?")]
    return " ".join(kept) if kept else headline


def score_articles(articles: list[dict], batch_size: int = 32, chunk: int = 256,
                   strip_questions: bool = True) -> list[dict]:
    device = 0 if torch.cuda.is_available() else -1  # GPU if present (e.g. Kaggle), else CPU
    print(f"loading {MODEL} on {'GPU' if device == 0 else 'CPU'} (first run downloads ~400 MB)...")
    clf = pipeline("text-classification", model=MODEL, top_k=None, device=device)

    for start in range(0, len(articles), chunk):
        batch = articles[start:start + chunk]
        texts = [scoring_text(a["headline"], strip_questions) for a in batch]
        results = clf(texts, batch_size=batch_size, truncation=True, max_length=128)
        for a, text, scores in zip(batch, texts, results):
            p = {d["label"].lower(): d["score"] for d in scores}
            a["sentiment"] = round(p.get("positive", 0.0) - p.get("negative", 0.0), 4)
            a["sentiment_label"] = max(p, key=p.get)
            a["sentiment_confidence"] = round(max(p.values()), 4)
            a["sentiment_text"] = text
        print(f"  scored {min(start + chunk, len(articles))}/{len(articles)}")
    return articles


def evaluate(articles: list[dict]) -> None:
    """Compare FinBERT against the human labels that came with the Kaggle dataset."""
    labeled = [a for a in articles if a.get("label") in ("positive", "negative", "neutral")]
    if not labeled:
        print("no labeled articles in this run, skipping evaluation")
        return
    from sklearn.metrics import classification_report, confusion_matrix
    y_true = [a["label"] for a in labeled]
    y_pred = [a["sentiment_label"] for a in labeled]
    names = ["negative", "neutral", "positive"]
    print(f"\n=== FinBERT vs human labels ({len(labeled)} headlines) ===")
    print(classification_report(y_true, y_pred, labels=names, digits=3))
    print("confusion matrix (rows = true, cols = predicted):", names)
    print(confusion_matrix(y_true, y_pred, labels=names))


def preview_live(articles: list[dict], n: int = 30) -> None:
    """Print the first n non-Kaggle (live) headlines with their scores."""
    live = [a for a in articles if a.get("source") != "kaggle_csv"]
    if not live:
        return
    print(f"\n{len(live)} live headlines\n")
    for a in live[:n]:
        print(f'{a["sentiment"]:+.2f}  [{a["source"][:14]}]  {a["headline"][:90]}')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/articles.jsonl")
    ap.add_argument("--output", default="data/scored.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="score only a random sample of N articles")
    ap.add_argument("--no-strip-questions", action="store_true",
                    help="score the full headline, including trailing questions")
    args = ap.parse_args()

    articles = load_jsonl(args.input)
    if args.limit and args.limit < len(articles):
        random.seed(42)
        articles = random.sample(articles, args.limit)

    scored = score_articles(articles, strip_questions=not args.no_strip_questions)
    write_jsonl(scored, args.output)
    evaluate(scored)
    preview_live(scored)