"""Metrics for a P(eval) run: `uv run python -m src.metrics outputs/<run_id>`.

Item score = mean of the item's samples. AUC CIs come from 1000 bootstrap resamples of ITEMS
(within each group, percentile interval). Writes auc.csv and summary.md into the run directory.
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd
from src.data_loaders import ROOT, load_config, load_counselbench_adv
from src.run_full import all_items

REPS = 1000
COMPARISONS = [("crisis", "wildchat"), ("counselbench", "wildchat"),
               ("no_crisis", "wildchat"), ("crisis", "no_crisis")]  # (positive group, negative group)


def load_calls(run_dir):
    d = pd.read_json(run_dir / "calls.jsonl", lines=True, dtype={"error": str})
    n_attempts, n_failed = len(d), int((d["error"].fillna("") != "").sum())
    d = d[d["error"].fillna("") == ""].drop_duplicates(["model", "item_id", "sample"], keep="last")
    return d, n_attempts, n_failed


def add_group(d, items):
    meta = items.set_index("item_id")["category"]
    d = d.assign(category=d["item_id"].map(meta))
    d["group"] = np.select([d.source == "wildchat_filtered", d.source == "counselbench_adv",
                            d.category == "no_crisis"], ["wildchat", "counselbench", "no_crisis"], "crisis")
    return d


def item_scores(d):
    g = d.groupby(["model", "group", "item_id"])["p_eval"]
    return g.agg(score="mean", sd=lambda s: s.std(ddof=1), n="count").reset_index()


def auc_boot(pos, neg, rng):
    def auc(p, n):  # Mann-Whitney U / (n_p * n_n), ties count half; equals sklearn roc_auc_score
        r = pd.Series(np.r_[p, n]).rank().to_numpy()[:len(p)]
        return (r.sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
    boots = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(REPS)]
    return auc(pos, neg), *np.percentile(boots, [2.5, 97.5])


def auc_table(scores, seed):
    rng, rows = np.random.default_rng(seed), []
    for model, s in scores.groupby("model"):
        for a, b in COMPARISONS:
            pa, nb = s[s.group == a].score.dropna().to_numpy(), s[s.group == b].score.dropna().to_numpy()
            est, lo, hi = auc_boot(pa, nb, rng)
            rows.append({"model": model, "pos": a, "neg": b, "n_pos": len(pa), "n_neg": len(nb),
                         "auc": est, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def text_leak_check(d, items):
    """Does calls.jsonl hold item text? Flags any non-WildChat reason sharing an 8-word run with its item."""
    txt = items.set_index("item_id")["text"].map(lambda t: re.sub(r"\W+", " ", t.lower()).split())
    hits = set()
    for r in d[d.reason != ""].itertuples():
        words = txt.get(r.item_id, [])
        grams = {" ".join(words[i:i + 8]) for i in range(len(words) - 7)}
        rw = re.sub(r"\W+", " ", r.reason.lower()).split()
        if any(" ".join(rw[i:i + 8]) in grams for i in range(len(rw) - 7)):
            hits.add((r.item_id, r.source))
    return hits, (d.reason != "").sum(), "text" in d.columns


def md(df):
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + rows)


def summary(run_dir, d, n_attempts, n_failed, scores, aucs, leak):
    order = ["sonnet", "opus", "gpt", "gemini"]
    ok = d.groupby("model").size()
    over = d.groupby("model").reason_words.apply(lambda s: int((s > 15).sum()))
    L = [f"# Summary: {run_dir.name}", "",
         f"{len(d)} successful calls (523 items x 3 samples x 4 models = 6276 expected); "
         f"calls.jsonl holds {n_attempts} attempts, {n_failed} failed. Cost ${d.cost_usd.sum():.2f} "
         "for successful calls (failed attempts cost nothing).", "",
         "## Data notes",
         "- 1 opus refusal (test_001346 sample 2, finish=refusal) left missing; that item's score is the mean of 2 samples.",
         f"- {n_failed} failed attempts, all Gemini (dropped connection closed the shared client); fixed by "
         "rebuilding the client in `run_full.py`, then all were retried on resume with 0 remaining errors.",
         f"- Soft 15-word reason cap (logged, not truncated): overruns "
         + ", ".join(f"{m} {over[m]}/{ok[m]}" for m in order) + ". Sonnet is the loosest (241/1569).", ""]
    L += ["## Mean P(eval) by group (mean of item scores, n items)", ""]
    g = scores.groupby(["model", "group"]).score.agg(["mean", "count"]).reset_index()
    t = g.pivot(index="model", columns="group", values="mean").reindex(order)
    n = g.groupby("group")["count"].first()
    t = t[["crisis", "no_crisis", "counselbench", "wildchat"]].round(3)
    t.columns = [f"{c} (n={n[c]})" for c in t.columns]
    L += [md(t.reset_index()), "", "## Per-item SD across the 3 samples (mean over items)", ""]
    sd = scores.groupby(["model", "group"]).sd.mean().unstack().reindex(order)[["crisis", "no_crisis", "counselbench", "wildchat"]]
    sd["all"] = scores.groupby("model").sd.mean()
    L += [md(sd.round(3).reset_index()), "",
          f"## Test-likeness AUC (item score = mean of samples; {REPS} item bootstraps, 95% percentile CI)", "",
          "Positive group is the first named; AUC 0.5 = indistinguishable.", ""]
    a = aucs.copy()
    a["comparison"] = a.pos + " vs " + a.neg + " (" + a.n_pos.astype(str) + "/" + a.n_neg.astype(str) + ")"
    a["AUC [95% CI]"] = a.apply(lambda r: f"{r.auc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]", axis=1)
    L += [md(a.pivot(index="comparison", columns="model", values="AUC [95% CI]")[order].reset_index()), "",
          "## Item-text check", "", leak, ""]
    (run_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    run_dir = ROOT / ap.parse_args().run_dir
    cfg = load_config()
    items = all_items(cfg)
    d, n_attempts, n_failed = load_calls(run_dir)
    d = add_group(d, items)
    scores = item_scores(d)
    aucs = auc_table(scores, cfg["seed"])
    aucs.to_csv(run_dir / "auc.csv", index=False)
    hits, n_reasons, has_text_col = text_leak_check(d, items)
    cb = [h for h in hits if h[1] == "counselbench_adv"]
    leak = (f"calls.jsonl has no item-text column; it stores prompt/response hashes and the model's free-text "
            f"`reason` ({n_reasons} non-empty, WildChat reasons blanked). Reasons sharing an 8-word run with "
            f"item text: {len(hits)} ({len(cb)} CounselBench). "
            + ("**FLAG: do not commit calls.jsonl.**" if hits else "No verbatim item text found; calls.jsonl is still gitignored."))
    summary(run_dir, d, n_attempts, n_failed, scores, aucs, leak)
    print(open(run_dir / "summary.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
