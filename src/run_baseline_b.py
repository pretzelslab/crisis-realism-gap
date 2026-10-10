"""P(eval) on Baseline B: `uv run python -m src.run_baseline_b [--run-id ID] [--limit N]`.
Same prompt, models and samples as run_full (4 models x 3 samples); reasons are stored (synthetic data).
Resumable per run id; aborts past COST_STOP. Output: outputs/<run_id>/calls.jsonl
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from src.data_loaders import ROOT, load_config
from src.run_full import ERR_STOP, WORKERS, one_call

COST_STOP = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cost-stop", type=float, default=COST_STOP)
    ap.add_argument("--items", default="data/synthetic/baseline_b.jsonl")
    ap.add_argument("--variant", help="keep only items whose `variant` field equals this (e.g. noask)")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    run_id = args.run_id or datetime.now().strftime("baselineB_peval_%Y%m%d_%H%M%S")
    out = ROOT / "outputs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "calls.jsonl"
    items = pd.read_json(ROOT / args.items, lines=True)
    if args.variant:
        items = items[items["variant"] == args.variant]
    if args.limit:
        items = items.head(args.limit)
    done, cost, n, nerr = set(), 0.0, 0, 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            cost += r["cost_usd"]
            if not r["error"]:
                done.add((r["model"], r["item_id"], r["sample"]))
    jobs = [(m, it, k) for m in cfg["models"] for _, it in items.iterrows()
            for k in range(cfg["elicitation"]["samples_per_item"]) if (m["name"], it["item_id"], k) not in done]
    print(f"{run_id}: {len(jobs)} calls, {len(done)} done, cost so far ${cost:.2f}", flush=True)
    lock, stop, f = threading.Lock(), threading.Event(), open(path, "a", encoding="utf-8")

    def work(job):
        nonlocal cost, n, nerr
        if stop.is_set():
            return
        r = one_call(*job)
        with lock:
            f.write(json.dumps(r) + "\n")
            f.flush()
            cost, n, nerr = cost + r["cost_usd"], n + 1, nerr + bool(r["error"])
            if n % 100 == 0:
                print(f"{n} calls, ${cost:.2f}, errors {nerr}", flush=True)
            if cost > args.cost_stop or (n >= 50 and nerr / n > ERR_STOP):
                stop.set()
                print(f"STOP: cost ${cost:.2f}, errors {nerr}/{n}", flush=True)

    with ThreadPoolExecutor(WORKERS * len(cfg["models"])) as ex:
        list(ex.map(work, jobs))
    f.close()
    print("finished" if not stop.is_set() else "aborted", f"calls={n} cost=${cost:.2f} errors={nerr}", flush=True)


if __name__ == "__main__":
    main()
