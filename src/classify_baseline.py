"""Baseline A precision filter: Haiku labels each WildChat candidate y/n/unsure.

Run: `uv run python -m src.classify_baseline`
Input : data/processed/wildchat_candidates.parquet (from build_wildchat_baseline.py)
Output: data/processed/wildchat_baseline.parquet (sampled y's), data/processed/wildchat_labels.parquet
        (all labels, local only), outputs/<run_id>/{calls.jsonl,summary.md} (hashes + counts, no text).
Message text is never printed or logged; the classifier's `reason` stays in gitignored data/processed/.
"""
import hashlib
import json
import re
from datetime import datetime

import anthropic
import pandas as pd
from dotenv import load_dotenv

from src.data_loaders import ROOT, load_config

PROMPT = """You are labelling messages sent to an AI chatbot. Answer two questions about the message.
Q1: Is this a real person talking about their own life or the life of someone close to them?
Q2: Are they seeking help, comfort or advice about it?
Label "y" only if BOTH are yes. Label "n" for stories, fiction, roleplay, tasks (writing, coding, translation), general knowledge questions, or users testing the bot. Label "unsure" if you cannot tell.
Reply with JSON only: {{"label": "y"|"n"|"unsure", "confidence": <0 to 1>, "reason": "<at most 15 words; describe, do not quote the message>"}}

Message:
<<<
{text}
>>>"""
MAX_OUT = 150


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0))
        label = str(d["label"]).strip().lower()
        conf = float(d["confidence"])
        if label in ("y", "n", "unsure") and 0 <= conf <= 1:
            return label, conf, str(d.get("reason", ""))
    except Exception:
        pass
    return None, None, ""


def main():
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    clf = cfg["baseline_classifier"]
    cand = pd.read_parquet(ROOT / "data/processed/wildchat_candidates.parquet")
    run_id = "baselineA_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / run_id
    out_dir.mkdir(parents=True)
    client = anthropic.Anthropic()

    rows, tin, tout = [], 0, 0
    with open(out_dir / "calls.jsonl", "w") as log:
        for it in cand.itertuples():
            prompt = PROMPT.format(text=it.text)
            r = client.messages.create(model=clf["id"], max_tokens=MAX_OUT,  # no temperature: SDK rejects it
                                       messages=[{"role": "user", "content": prompt}])
            raw = "".join(b.text for b in r.content if b.type == "text")
            label, conf, reason = parse(raw)
            tin += r.usage.input_tokens
            tout += r.usage.output_tokens
            log.write(json.dumps({"item_id": it.item_id, "prompt_sha": sha(prompt), "response_sha": sha(raw),
                                  "model": r.model, "label": label, "confidence": conf}) + "\n")
            rows.append({"item_id": it.item_id, "label": label, "confidence": conf, "reason": reason})

    labels = pd.DataFrame(rows)
    labels.to_parquet(ROOT / "data/processed/wildchat_labels.parquet")
    merged = cand.merge(labels, on="item_id", validate="1:1")
    keep = merged[(merged["label"] == "y") & (merged["confidence"] >= clf["min_confidence"])]
    n = min(len(keep), cfg["sample_sizes"]["deployment_baseline"])
    out = keep.sample(n=n, random_state=cfg["seed"]).reset_index(drop=True)
    out["label_origin"] = f"{clf['id']} y>={clf['min_confidence']}"
    out.to_parquet(ROOT / "data/processed/wildchat_baseline.parquet")

    cost = tin / 1e6 * clf["price_in"] + tout / 1e6 * clf["price_out"]
    counts = labels["label"].fillna("unparsed").value_counts().to_dict()
    summary = (f"# {run_id}\n\nModel: {clf['id']} (default sampling; SDK has no temperature param)\n\n"
               f"- Candidates classified: {len(cand)}\n- Labels: {counts}\n"
               f"- y with confidence >= {clf['min_confidence']}: {len(keep)}\n"
               f"- Sampled into baseline: {n} (seed {cfg['seed']})\n"
               f"- Tokens in/out: {tin}/{tout}; est. cost ${cost:.3f}\n")
    (out_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
