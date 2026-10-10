"""Baseline A-crisis, step 1: crisis-keyword prefilter over WildChat shards 7-13 (unused by Baseline A).

Standalone (run with `.venv/Scripts/python -I`): the parquet files are untrusted downloads. Prints aggregate counts only;
never prints message text. Output stays in data/processed/ (gitignored).
Filter: English, not toxic/redacted, single-turn conversation (exactly one user turn), first user turn 30-1500 chars,
first-person pronoun + a crisis keyword (six benchmark categories), no task-like keywords (as Baseline A), text-deduped,
and excluded if the text is already in Baseline A (candidates or sample).
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SHARDS = range(7, 14)

FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|ive|i'll|my|me|myself)\b", re.I)
TASKLIKE = re.compile(
    r"\b(write|essay|code|python|javascript|translate|summari[sz]e|rewrite|paraphrase|"
    r"generate|prompt|story|poem|script|article|email|resume|cover letter|explain|"
    r"midjourney|stable diffusion|list of|table|function|sql|html|css)\b", re.I)
KEYWORDS = {
    "anxiety_crisis": r"panic attacks?|anxiety attacks?|can'?t breathe|heart (is )?racing|can'?t stop worrying|so anxious|"
                      r"overwhelming anxiety|spiral(l)?ing|freaking out|losing (my mind|it)",
    "self_harm": r"self[- ]?harm\w*|cut(ting)? myself|hurt(ing)? myself|harm(ing)? myself|burn(ing)? myself|urge to hurt",
    "substance_abuse_or_withdrawal": r"addicted|addiction|withdrawals?|relapse[ds]?|drinking too much|can'?t stop drinking|"
                                     r"drunk (every|all)|high every|using again|can'?t quit (drinking|smoking|weed|drugs)|alcoholic",
    "suicidal_ideation": r"suicid\w*|kill myself|end my life|want to die|don'?t want to (live|be alive|be here)|"
                         r"better off dead|end it all|not be alive|wish i was(n'?t)? (dead|alive)|take my own life",
    "violent_thoughts": r"want to kill (him|her|them|someone|my|a)|(hurt|punch|hit|strangle|stab) (him|her|them|someone|my)|"
                        r"violent (thoughts|urges)|so angry i|homicidal|rage",
    "risk_taking_behaviors": r"reckless\w*|drive drunk|drunk driving|dangerous things|can'?t stop gambling|gambling problem|"
                             r"risky behaviou?r|unsafe sex|self[- ]destruct\w*",
}
KW = {k: re.compile(r"\b(" + v + r")\b", re.I) for k, v in KEYWORDS.items()}


def first_user_turn(conv):
    users = [u for u in conv if u.get("role") == "user"]
    return users[0] if len(users) == 1 else None  # single-turn only


def main():
    base = pd.read_parquet(ROOT / "data/processed/wildchat_candidates.parquet", columns=["item_id", "text"])
    seen_ids = set(base["item_id"])
    seen_text = set(base["text"])
    files = [ROOT / f"data/raw/wildchat_dl/data/train-{i:05d}-of-00014.parquet" for i in SHARDS]
    stages = {"rows": 0, "english": 0, "clean_flags": 0, "single_turn_length_ok": 0, "task_excluded": 0,
              "first_person": 0, "keyword_hit": 0, "dropped_in_baseline_A": 0}
    per_cat = {k: 0 for k in KW}
    keep = []
    for f in files:
        df = pd.read_parquet(f, columns=["conversation_hash", "language", "toxic", "redacted", "conversation"])
        stages["rows"] += len(df)
        df = df[df["language"] == "English"]
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
            stages["single_turn_length_ok"] += 1
            if TASKLIKE.search(t):
                stages["task_excluded"] += 1
                continue
            if not FIRST_PERSON.search(t):
                continue
            stages["first_person"] += 1
            hits = [k for k, rx in KW.items() if rx.search(t)]
            if not hits:
                continue
            stages["keyword_hit"] += 1
            iid = f"wc_{h[:12]}"
            if iid in seen_ids or t in seen_text:
                stages["dropped_in_baseline_A"] += 1
                continue
            for k in hits:
                per_cat[k] += 1
            keep.append({"shard": f.name[6:11], "item_id": iid, "source": "wildchat_filtered", "n_turns": 1, "text": t,
                         "text_first_turn": t, "category": "wildchat_crisis_candidate", "label_origin": "keyword",
                         "keyword_cats": ",".join(hits)})
    cand = pd.DataFrame(keep).drop_duplicates("text").reset_index(drop=True)
    stages["after_dedupe"] = len(cand)
    print(stages)
    print("keyword-hit categories (an item can hit several):", per_cat)
    print("hit rate = keyword_hit / single_turn_length_ok_minus_tasks = "
          f"{stages['keyword_hit'] / (stages['single_turn_length_ok'] - stages['task_excluded']):.5f}; "
          f"/ English clean rows = {stages['keyword_hit'] / stages['clean_flags']:.5f}")
    dest = ROOT / "data/processed/wildchat_crisis_candidates.parquet"
    cand.to_parquet(dest)
    print("wrote", dest.name, len(cand), "per shard:", cand["shard"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
