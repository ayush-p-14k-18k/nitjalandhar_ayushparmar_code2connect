import json
rows = [json.loads(l) for l in open("data/scored.jsonl", encoding="utf-8")]
live = [r for r in rows if r["source"] != "kaggle_csv"]
print(len(live), "live headlines\n")
for r in live[:30]:
    print(f'{r["sentiment"]:+.2f}  [{r["source"][:14]}]  {r["headline"][:90]}')