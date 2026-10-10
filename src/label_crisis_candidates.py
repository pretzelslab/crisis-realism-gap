"""Baseline A-crisis, step 2: Haiku crisis_level labels on keyword candidates, in seeded random order, until
STOP_YES crisis-level items are found or COST_STOP is reached. Same prompt as src/label_wildchat_crisis.py.
`uv run python -m src.label_crisis_candidates [--run-id ID]`  (resumable from calls.jsonl)
Output: data/processed/wildchat_crisis_labels_new.parquet + wildchat_crisis_real.parquet (crisis-level items, with text,
gitignored), data/processed/wildchat_crisis_review.csv (10 random crisis-level items, gitignored: real users' text),
outputs/<run_id>/{calls.jsonl,prompt.txt,summary.md} (ids/hashes/labels only; no text).
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import anthropic
import pandas as pd
from dotenv import load_dotenv

from src.data_loaders import ROOT, load_config
from src.label_wildchat_crisis import PROMPT, parse, sha

STOP_YES, COST_STOP, CHUNK = 60, 3.0, 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    clf = cfg["baseline_classifier"]
    run_id = ap.parse_args().run_id or "wildchat_crisis_new_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "outputs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    cand = pd.read_parquet(ROOT / "data/processed/wildchat_crisis_candidates.parquet")
    order = cand.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    log_path = out / "calls.jsonl"
    done, cost = {}, 0.0
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            cost += r["cost_usd"]
            if r["crisis_level"] is not None:
                done[r["item_id"]] = r
    client = anthropic.Anthropic()

    def label(it):
        p = PROMPT.format(text=it.text)
        r = client.messages.create(model=clf["id"], max_tokens=100, messages=[{"role": "user", "content": p}])
        raw = "".join(b.text for b in r.content if b.type == "text")
        lab, cat, conf = parse(raw)
        c = (r.usage.input_tokens * clf["price_in"] + r.usage.output_tokens * clf["price_out"]) / 1e6
        return {"item_id": it.item_id, "prompt_sha": sha(p), "response_sha": sha(raw), "model": r.model,
                "crisis_level": lab, "category": cat, "confidence": conf, "cost_usd": c}

    stop_reason = "exhausted candidates"
    n_yes = lambda: sum(r["crisis_level"] == "yes" for r in done.values())
    pos = 0
    with open(log_path, "a") as log, ThreadPoolExecutor(8) as ex:
        while pos < len(order):
            if n_yes() >= STOP_YES:
                stop_reason = f"reached {STOP_YES} crisis-level items"
                break
            if cost >= COST_STOP:
                stop_reason = f"cost cap ${COST_STOP}"
                break
            chunk = [it for it in order.iloc[pos:pos + CHUNK].itertuples() if it.item_id not in done]
            pos += CHUNK
            for r in ex.map(label, chunk):
                log.write(json.dumps(r) + "\n")
                log.flush()
                cost += r["cost_usd"]
                if r["crisis_level"] is not None:
                    done[r["item_id"]] = r
            print(f"{len(done)} labelled, {n_yes()} crisis-level, ${cost:.2f}", flush=True)

    lab = pd.DataFrame(done.values())[["item_id", "crisis_level", "category", "confidence"]]
    lab.to_parquet(ROOT / "data/processed/wildchat_crisis_labels_new.parquet")
    yes = cand.merge(lab[lab.crisis_level == "yes"], on="item_id", validate="1:1", suffixes=("_kw", ""))
    yes = yes.drop(columns=["category_kw"])  # placeholder category from the candidate file; "category" is now the Haiku label
    yes["label_origin"] = f"{clf['id']} crisis_level=yes"
    yes.to_parquet(ROOT / "data/processed/wildchat_crisis_real.parquet")
    rev = yes.sample(n=min(10, len(yes)), random_state=cfg["seed"])[["item_id", "text", "category", "confidence"]]
    rev = rev.rename(columns={"item_id": "id"})
    for c in ("topic_ok", "real_crisis", "notes"):
        rev[c] = ""
    rev.to_csv(ROOT / "data/processed/wildchat_crisis_review.csv", index=False, encoding="utf-8-sig")

    cats = yes.category.value_counts().to_dict()
    hi = int((yes.confidence >= 0.8).sum())
    L = [f"# {run_id}", "", f"Stop: {stop_reason}. Candidates available {len(cand)}, labelled {len(lab)}, cost ${cost:.2f} "
         f"(cap ${COST_STOP}). Model {clf['id']}, default sampling.",
         f"Crisis-level: {len(yes)} ({hi} with confidence >= 0.8). Precision of keyword prefilter (crisis-level / labelled): "
         f"{len(yes) / len(lab):.1%}.", f"By category: {cats}", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
