"""
pipeline.py - runs every stage of the risk engine in order.

    ingest.py     -> data/articles.jsonl
    sentiment.py  -> data/scored.jsonl
    events.py     -> data/classified.jsonl
    impact.py     -> data/signals.jsonl      (what api.py serves)

Usage:
    python pipeline.py                    # full run
    python pipeline.py --no-gdelt         # full run, skip GDELT (rate limited)
    python pipeline.py --from events      # re-run events + impact only (e.g. after a rule change)
    python pipeline.py --only impact      # re-run a single stage
    python pipeline.py --dry-run          # show what would run

Tip: after changing event rules or thresholds, `--from events` takes seconds because
zero-shot results are cached. `sentiment` is the slow stage (minutes on CPU).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [  # (name, script, input file it needs, output file it writes)
    ("ingest", "ingest.py", None, "data/articles.jsonl"),
    ("sentiment", "sentiment.py", "data/articles.jsonl", "data/scored.jsonl"),
    ("events", "events.py", "data/scored.jsonl", "data/classified.jsonl"),
    ("impact", "impact.py", "data/classified.jsonl", "data/signals.jsonl"),
]
NAMES = [s[0] for s in STAGES]


def count_lines(path: Path) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=NAMES, default="ingest",
                    help="start at this stage and run everything after it")
    ap.add_argument("--only", choices=NAMES, help="run just this stage")
    ap.add_argument("--no-gdelt", action="store_true", help="pass --no-gdelt to ingest.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    selected = [s for s in STAGES if s[0] == args.only] if args.only else \
               STAGES[NAMES.index(args.start):]

    # Fail early if the first selected stage has no input yet.
    need = selected[0][2]
    if need and not (ROOT / need).exists() and not args.dry_run:
        print(f"cannot start at '{selected[0][0]}': {need} does not exist yet. "
              f"Run an earlier stage first (e.g. `python pipeline.py --from ingest`).")
        return 2

    results = []
    for name, script, _, out_file in selected:
        cmd = [sys.executable, str(ROOT / script)]
        if name == "ingest" and args.no_gdelt:
            cmd.append("--no-gdelt")
        print(f"\n=== [{name}] {' '.join(cmd[1:])} ===", flush=True)
        if args.dry_run:
            results.append((name, "dry-run", 0.0, 0))
            continue
        t0 = time.time()
        code = subprocess.run(cmd, cwd=ROOT).returncode
        secs = time.time() - t0
        rows = count_lines(ROOT / out_file)
        results.append((name, "ok" if code == 0 else f"FAILED (exit {code})", secs, rows))
        if code != 0:
            print(f"\nstage '{name}' failed; stopping. Later stages were NOT run, "
                  f"so downstream files may be stale.")
            break

    print("\n--- summary ---")
    for name, status, secs, rows in results:
        print(f"{name:<10}{status:<18}{secs:7.1f}s   {rows} rows")
    ok = all(r[1] in ("ok", "dry-run") for r in results)
    if ok and not args.dry_run and selected[-1][0] == "impact":
        print("\nsignals ready: data/signals.jsonl")
        print("start the API:  uvicorn api:app --reload")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())