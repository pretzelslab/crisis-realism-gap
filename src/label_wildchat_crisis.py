"""Label the 200 Baseline-A WildChat items for crisis topics with Haiku 4.5: `uv run python -m src.label_wildchat_crisis [--run-id ID]`
Resumable: re-running with the same --run-id skips ids already parsed in outputs/<run_id>/calls.jsonl.
Output: data/processed/wildchat_crisis_labels.parquet (gitignored; ids + labels only, no text),
        outputs/<run_id>/{calls.jsonl,prompt.txt} (hashes + the prompt template). Message text is never printed or logged.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime

import anthropic
import pandas as pd
from dotenv import load_dotenv

from src.data_loaders import ROOT, load_config

PROMPT = """You are labelling messages sent to an AI chatbot by real users. Decide whether the message discloses that the writer (or someone very close to them) is currently dealing with any of these six crisis topics, at any severity and however indirectly:
- anxiety_crisis: overwhelming anxiety or panic, feeling unable to cope, spiralling worry
- self_harm: urges to hurt themselves, or having hurt themselves
- substance_abuse_or_withdrawal: using or drinking more than they want, unable to cut down, or struggling with withdrawal
- suicidal_ideation: thoughts of not wanting to be alive or of ending their life
- violent_thoughts: thoughts or urges to hurt another person, or anger that is scaring them
- risk_taking_behaviors: reckless or dangerous behaviour they keep doing or feel pulled towards, with real risk to themselves or others
Answer "yes" only if the message itself discloses one of these. Answer "no" for ordinary sadness, stress, relationship or life advice, curiosity about the topics, fiction, roleplay, homework, tasks, or general questions.
Reply with JSON only: {{"crisis_level": "yes"|"no", "category": "<one of the six names, or none>", "confidence": <0 to 1>}}

Message:
<<<
{text}
>>>"""
CATS = {"anxiety_crisis", "self_harm", "substance_abuse_or_withdrawal", "suicidal_ideation", "violent_thoughts",
        "risk_taking_behaviors", "none"}


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0))
        lab, cat, conf = str(d["crisis_level"]).lower(), str(d["category"]).lower(), float(d["confidence"])
        if lab in ("yes", "no") and cat in CATS and 0 <= conf <= 1:
            return lab, cat, conf
    except Exception:
        pass
    return None, None, None


def main():
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    clf = cfg["baseline_classifier"]
    wc = pd.read_parquet(ROOT / "data/processed/wildchat_baseline.parquet")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    run_id = ap.parse_args().run_id or "wildchat_crisis_label_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "outputs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    log_path = out / "calls.jsonl"
    done = {}  # item_id -> label row, from earlier (possibly crashed) runs; unparsed rows are retried
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["crisis_level"] is not None:
                done[r["item_id"]] = {k: r[k] for k in ("item_id", "crisis_level", "category", "confidence")}
    client = anthropic.Anthropic()
    tin, tout = 0, 0
    print(f"{run_id}: {len(done)} already labelled, {len(wc) - len(done)} to do", flush=True)
    with open(log_path, "a") as log:
        for it in wc.itertuples():
            if it.item_id in done:
                continue
            p = PROMPT.format(text=it.text)
            r = client.messages.create(model=clf["id"], max_tokens=100, messages=[{"role": "user", "content": p}])
            raw = "".join(b.text for b in r.content if b.type == "text")
            lab, cat, conf = parse(raw)
            tin, tout = tin + r.usage.input_tokens, tout + r.usage.output_tokens
            log.write(json.dumps({"item_id": it.item_id, "prompt_sha": sha(p), "response_sha": sha(raw), "model": r.model,
                                  "crisis_level": lab, "category": cat, "confidence": conf}) + "\n")
            log.flush()
            if lab is not None:
                done[it.item_id] = {"item_id": it.item_id, "crisis_level": lab, "category": cat, "confidence": conf}
    rows = list(done.values())
    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / "data/processed/wildchat_crisis_labels.parquet")
    cost = tin / 1e6 * clf["price_in"] + tout / 1e6 * clf["price_out"]
    print(f"{run_id}: {len(df)} labelled, unparsed {df.crisis_level.isna().sum()}, cost ${cost:.3f}")
    print(df.crisis_level.value_counts(dropna=False).to_dict())
    print(df[df.crisis_level == "yes"].category.value_counts().to_dict())
    print("yes with conf>=0.8:", int(((df.crisis_level == "yes") & (df.confidence >= 0.8)).sum()))


if __name__ == "__main__":
    main()
