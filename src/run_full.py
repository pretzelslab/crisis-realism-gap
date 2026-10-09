"""Full P(eval) run: `uv run python -m src.run_full [--run-id ID]`. Appends one JSON line per call to
outputs/<run_id>/calls.jsonl; re-running with the same --run-id skips finished calls (crash resume).
Aborts if cumulative cost passes COST_STOP or the error rate passes ERR_STOP (after 50 calls).
"""
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from src.classify_eval import PROMPT, call, parse, sha
from src.data_loaders import (ROOT, load_config, load_counselbench_adv,
                              load_crisis_benchmark, sample_crisis_benchmark)

COST_STOP, ERR_STOP, WORKERS = 15.0, 0.02, 4


def all_items(cfg):
    wild = pd.read_parquet(ROOT / "data/processed/wildchat_baseline.parquet")
    crisis = sample_crisis_benchmark(load_crisis_benchmark(), cfg["seed"])
    return pd.concat([crisis, load_counselbench_adv(), wild], ignore_index=True)


def one_call(model, it, k):
    prompt = PROMPT.format(text=it["text"])
    t0 = time.time()
    for attempt in range(6):
        try:
            raw, ret, tin, tout, fin, thought = call(model, prompt)
            err = ""
            break
        except Exception as e:
            raw, ret, tin, tout, fin, thought, err = "", "", 0, 0, "", 0, f"{type(e).__name__}: {str(e)[:200]}"
            if "client has been closed" in str(e) or "Server disconnected" in str(e):
                import src.classify_eval as ce
                ce._GEMINI = None  # rebuild the shared Gemini client after a dropped connection
            if not re.search(r"(429|500|503|529)|UNAVAILABLE|overloaded|rate|client has been closed|Server disconnected", str(e), re.I):
                break
            time.sleep(2 ** attempt)
    p, reason = parse(raw)
    wild = it["source"] == "wildchat_filtered"
    return {"model": model["name"], "model_id": model["id"], "returned_model": ret,
            "item_id": it["item_id"], "source": it["source"], "sample": k,
            "prompt_hash": sha(prompt), "response_hash": sha(raw), "p_eval": p,
            "reason": "" if wild else reason, "reason_words": len(reason.split()),
            "tokens_in": tin, "tokens_out": tout, "thought_tokens": thought, "finish": fin,
            "cost_usd": (tin * model["price_in"] + tout * model["price_out"]) / 1e6,
            "latency_s": round(time.time() - t0, 2), "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    run_id = args.run_id or datetime.now().strftime("full_%Y%m%d_%H%M%S")
    out = ROOT / "outputs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "calls.jsonl"
    done, cost, n, nerr = set(), 0.0, 0, 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["error"]:
                continue  # retry failed calls on resume
            done.add((r["model"], r["item_id"], r["sample"]))
            cost += r["cost_usd"]
    items = all_items(cfg).sample(frac=1, random_state=cfg["seed"])
    jobs = [(m, it, k) for m in cfg["models"] for _, it in items.iterrows()
            for k in range(cfg["elicitation"]["samples_per_item"])
            if (m["name"], it["item_id"], k) not in done]
    print(f"{run_id}: {len(jobs)} calls to do, {len(done)} already done, cost so far ${cost:.2f}", flush=True)
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
            if cost > COST_STOP or (n >= 50 and nerr / n > ERR_STOP):
                stop.set()
                print(f"STOP: cost ${cost:.2f}, errors {nerr}/{n}", flush=True)

    with ThreadPoolExecutor(WORKERS * len(cfg["models"])) as ex:
        list(ex.map(work, jobs))
    f.close()
    print("finished" if not stop.is_set() else "aborted", f"calls={n} cost=${cost:.2f} errors={nerr}", flush=True)


if __name__ == "__main__":
    main()
