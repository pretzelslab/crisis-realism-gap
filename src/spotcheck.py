"""Local spot-check CLI: show 20 random classifier-'y' WildChat messages with the model's reason;
you answer agree/disagree. Run: `uv run python -m src.spotcheck [--n 20] [--seed N]`.

Message text is shown only in your own terminal and is never written to the saved results
(item_id, verdict, confidence only). Results: outputs/spotcheck/<timestamp>.csv + precision printed.
"""
import argparse
import textwrap
from datetime import datetime

import pandas as pd

from src.data_loaders import ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None, help="default: config seed")
    a = ap.parse_args()
    cfg = load_config()
    thr = cfg["baseline_classifier"]["min_confidence"]

    cand = pd.read_parquet(ROOT / "data/processed/wildchat_candidates.parquet")
    labels = pd.read_parquet(ROOT / "data/processed/wildchat_labels.parquet")
    ys = cand.merge(labels, on="item_id")
    ys = ys[(ys["label"] == "y") & (ys["confidence"] >= thr)]
    sample = ys.sample(n=min(a.n, len(ys)), random_state=cfg["seed"] if a.seed is None else a.seed)

    results = []
    for i, r in enumerate(sample.itertuples(), 1):
        print(f"\n{'=' * 70}\n[{i}/{len(sample)}]  confidence {r.confidence:.2f}\n")
        print(textwrap.fill(r.text, 100, replace_whitespace=False))
        print(f"\nModel's reason: {r.reason}")
        while True:
            ans = input("agree / disagree / quit  [a/d/q]: ").strip().lower()
            if ans in ("a", "d", "q"):
                break
        if ans == "q":
            break
        results.append({"item_id": r.item_id, "verdict": "agree" if ans == "a" else "disagree",
                        "confidence": r.confidence})

    if not results:
        print("\nNothing saved.")
        return
    out = pd.DataFrame(results)
    dest = ROOT / "outputs" / "spotcheck"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.to_csv(path, index=False)
    agree = (out["verdict"] == "agree").sum()
    print(f"\nSaved {len(out)} verdicts to {path.relative_to(ROOT)}")
    print(f"Agreement (estimated precision of y>={thr}): {agree}/{len(out)} = {agree / len(out):.0%}")


if __name__ == "__main__":
    main()
