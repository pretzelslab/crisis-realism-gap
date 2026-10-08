"""Baseline A: filter WildChat-1M to English support-seeking first user turns, sample n.

Standalone (run with `python -I`): the parquet files are untrusted downloads. Prints
aggregate counts only; never prints message text. Output stays in data/processed/ (gitignored).
"""
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|ive|i'll|my|me|myself)\b", re.I)
EMOTION = re.compile(
    r"\b(feel(ing)?|felt|sad|depress\w*|anxi\w*|lonely|alone|stress\w*|scared|afraid|worried|"
    r"overwhelm\w*|struggl\w*|hopeless|exhausted|tired of|cry(ing)?|panic\w*|hurt(ing)?|"
    r"heartbroken|broke up|breakup|lost (my|a)|grief|miss(ing)? my|can'?t sleep|"
    r"don'?t know what to do|need (some )?(advice|help|someone)|what should i do|"
    r"toxic|abusive|argument|fight with|my (boyfriend|girlfriend|husband|wife|partner|ex|"
    r"mom|mother|dad|father|parents|friend|friends|boss|therapist|family))\b", re.I)
TASKLIKE = re.compile(
    r"\b(write|essay|code|python|javascript|translate|summari[sz]e|rewrite|paraphrase|"
    r"generate|prompt|story|poem|script|article|email|resume|cover letter|explain|"
    r"midjourney|stable diffusion|list of|table|function|sql|html|css)\b", re.I)


def first_user_turn(conv):
    for u in conv:
        if u.get("role") == "user":
            return u
    return None


def main(n_sample):
    files = sorted((ROOT / "data/raw/wildchat").glob("train-*.parquet"))
    stages = {"rows": 0, "english": 0, "clean_flags": 0, "length_ok": 0,
              "emotional_not_task": 0}
    keep = []
    for f in files:
        df = pd.read_parquet(f, columns=["conversation_hash", "language", "toxic",
                                         "redacted", "conversation"])
        stages["rows"] += len(df)
        df = df[(df["language"] == "English")]
        stages["english"] += len(df)
        df = df[(~df["toxic"].astype(bool)) & (~df["redacted"].astype(bool))]
        stages["clean_flags"] += len(df)
        for h, conv in zip(df["conversation_hash"], df["conversation"]):
            u = first_user_turn(conv)
            if u is None or u.get("toxic") or u.get("redacted"):
                continue
            t = (u.get("content") or "").strip()
            if not (30 <= len(t) <= 1500):
                continue
            stages["length_ok"] += 1
            if FIRST_PERSON.search(t) and EMOTION.search(t) and not TASKLIKE.search(t):
                keep.append({"item_id": f"wc_{h[:12]}", "source": "wildchat_filtered",
                             "n_turns": 1, "text": t, "text_first_turn": t,
                             "category": "deployment_baseline", "label_origin": "heuristic"})
    stages["emotional_not_task"] = len(keep)
    cand = pd.DataFrame(keep).drop_duplicates("text")
    stages["after_dedupe"] = len(cand)
    print(stages)
    if len(cand) < n_sample:
        print(f"only {len(cand)} candidates; need more shards")
        sys.exit(2)
    out = cand.sample(n=n_sample, random_state=cfg["seed"]).reset_index(drop=True)
    dest = ROOT / "data/processed/wildchat_baseline.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest)
    print("wrote", dest.name, len(out), "chars median", int(out["text"].str.len().median()),
          "chars max", int(out["text"].str.len().max()))


if __name__ == "__main__":
    main(cfg["sample_sizes"]["deployment_baseline"])
