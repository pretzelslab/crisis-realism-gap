"""Baseline C analysis: `uv run python -m src.metrics_c outputs/<baselineC_peval_run>`.
Groups: C (163 synthetic non-crisis), B_noask (163: 93 unchanged B items + 70 no-ask regenerations), wildchat (200),
no_crisis (benchmark no_crisis controls), plus bench_single/bench_multi/B for reference. Item score = mean of samples; 1000 item bootstraps.
Appends to <run_dir>/summary.md.
"""
import argparse

import numpy as np
import pandas as pd

from src.data_loaders import ROOT, load_config
from src.metrics import add_group, auc_boot, load_calls, md
from src.metrics_b import ORDER, fmt, strat_boot
from src.run_full import all_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--b-run", default="outputs/baselineB_peval_20261010")
    ap.add_argument("--noask-run", default="outputs/baselineB_noask_peval_20261010")
    ap.add_argument("--phase1", default="outputs/full_20261008")
    a = ap.parse_args()
    run_dir, cfg = ROOT / a.run_dir, load_config()
    rng = np.random.default_rng(cfg["seed"])

    items = all_items(cfg)
    d1, _, _ = load_calls(ROOT / a.phase1)
    d1 = add_group(d1, items)
    meta = items.set_index("item_id")
    d1["turns"] = d1.item_id.map(meta.n_turns)
    d1["words"] = d1.item_id.map(meta.text.str.split().str.len())
    d1["grp"] = np.where(d1.group == "wildchat", "wildchat", np.where(d1.group == "no_crisis", "no_crisis",
                         np.where(d1.group == "crisis", np.where(d1.turns == 1, "bench_single", "bench_multi"), "other")))
    s1 = d1[d1.grp != "other"].groupby(["model", "grp", "item_id"]).agg(score=("p_eval", "mean"), words=("words", "first")).reset_index()
    nc_turns = meta.loc[meta.index.isin(s1[s1.grp == "no_crisis"].item_id), "n_turns"]

    orig = pd.read_json(ROOT / "data/synthetic/baseline_b.jsonl", lines=True)
    na = pd.read_json(ROOT / "data/synthetic/baseline_b_noask.jsonl", lines=True)
    cc = pd.read_json(ROOT / "data/synthetic/baseline_c.jsonl", lines=True)
    dO, _, _ = load_calls(ROOT / a.b_run)
    dN, _, _ = load_calls(ROOT / a.noask_run)
    dC, n_att, n_fail = load_calls(run_dir)
    sO = dO.groupby(["model", "item_id"]).p_eval.mean().rename("score").reset_index()
    sN = dN.groupby(["model", "item_id"]).p_eval.mean().rename("score").reset_index()
    new_ids = set(sN.item_id)
    sNA = pd.concat([sN, sO[~sO.item_id.isin(new_ids)]]).merge(na[["item_id", "words"]], on="item_id").assign(grp="B_noask")
    sB = sO.merge(orig[["item_id", "words"]], on="item_id").assign(grp="B")
    sC = dC.groupby(["model", "item_id"]).p_eval.mean().rename("score").reset_index().merge(cc[["item_id", "words", "topic"]], on="item_id").assign(grp="C")
    assert sC.groupby("model").item_id.nunique().eq(163).all() and sNA.groupby("model").item_id.nunique().eq(163).all()
    cols = ["model", "grp", "item_id", "score", "words"]
    allsc = pd.concat([s1, sNA[cols], sB[cols], sC[cols]], ignore_index=True)

    wild_w = allsc[(allsc.model == "sonnet") & (allsc.grp == "wildchat")].words
    edges = np.quantile(np.log(wild_w), [.2, .4, .6, .8])
    comps = [("C", "wildchat"), ("B_noask", "C"), ("no_crisis", "C"),
             ("no_crisis", "wildchat"), ("B_noask", "wildchat"), ("bench_single", "C")]  # last three = references
    n = {g: allsc[(allsc.model == "sonnet") & (allsc.grp == g)].shape[0] for g in allsc.grp.unique()}
    rows = []
    for model in ORDER:
        m = allsc[allsc.model == model]
        for p, q in comps:
            P, N = m[m.grp == p], m[m.grp == q]
            x, y = P.score.to_numpy(), N.score.to_numpy()
            rows.append({"comparison": f"{p} ({len(x)}) vs {q} ({len(y)})", "model": model,
                         "AUC [95% CI]": fmt(auc_boot(x, y, rng)),
                         "length-matched": fmt(strat_boot(x, y, np.log(P.words.to_numpy()), np.log(N.words.to_numpy()), edges, rng))})
    auc_df = pd.DataFrame(rows)
    order = [f"{p} ({n[p]}) vs {q} ({n[q]})" for p, q in comps]

    def piv(col):
        return auc_df.pivot(index="comparison", columns="model", values=col)[ORDER].reindex(order).reset_index()

    mean_group = allsc.groupby(["model", "grp"]).score.mean().unstack().reindex(ORDER)[["bench_single", "bench_multi", "no_crisis", "B", "B_noask", "C", "wildchat"]]
    mean_group.columns = [f"{c} (n={n[c]})" for c in mean_group.columns]
    topic = sC.groupby(["model", "topic"]).score.mean().unstack().reindex(ORDER)
    topic["all"] = sC.groupby("model").score.mean()
    topic_n = cc.topic.value_counts()
    topic.columns = [f"{c} ({topic_n[c]})" if c in topic_n else c for c in topic.columns]
    ok = dC.groupby("model").size().reindex(ORDER)
    over = dC.groupby("model").reason_words.apply(lambda s: int((s > 15).sum())).reindex(ORDER)
    sd = dC.groupby(["model", "item_id"]).p_eval.std(ddof=1).groupby("model").mean().reindex(ORDER).round(3)

    L = ["", "## Baseline C", "",
         f"Synthetic non-crisis, realistic: 163 single-turn items (Haiku 4.5), same style spec as B-noask (no unrelated ask), same length targets as Baseline B, "
         f"topic mix sampled from Haiku topic labels of the 200 WildChat items (`data/synthetic/baseline_c_topic_mix.json`), Haiku crisis-content screen (reject yes). "
         f"Generation and checks: `outputs/{sorted((ROOT / 'outputs').glob('baselineC_2*'))[-1].name}/summary.md`. "
         f"P(eval): run `{run_dir.name}`, {len(dC)} successful calls of 1956 expected ({n_att} attempts, {n_fail} failed), cost ${dC.cost_usd.sum():.2f} (cap $3), "
         f"same prompt, models and 3 samples as before; reasons stored (synthetic). "
         f"Soft 15-word reason cap overruns: " + ", ".join(f"{m} {over[m]}/{ok[m]}" for m in ORDER) + ". Per-item SD across samples: " + ", ".join(f"{m} {sd[m]}" for m in ORDER) + ".",
         f"Bench no_crisis controls: {len(nc_turns)} items, {int((nc_turns == 1).sum())} single-turn and {int((nc_turns > 1).sum())} multi-turn.", "",
         "### Mean P(eval) by group (mean of item scores)", "", md(mean_group.round(3).reset_index()), "",
         "### Test-likeness AUC, raw (first-named group is positive; 1000 item bootstraps, 95% percentile CI)", "", md(piv("AUC [95% CI]")), "",
         "### Test-likeness AUC, length-matched (WildChat length quintiles, log words)", "", md(piv("length-matched")), "",
         "Rows 4 to 6 are references (not requested comparisons).", "",
         "### Mean P(eval) in C by topic (n items in brackets)", "", md(topic.round(3).reset_index()), ""]
    with open(run_dir / "summary.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))
    auc_df.to_csv(run_dir / "auc.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
