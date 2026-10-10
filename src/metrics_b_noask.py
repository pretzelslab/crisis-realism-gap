"""B-noask analysis: `uv run python -m src.metrics_b_noask outputs/<noask_peval_run> [--b-run outputs/baselineB_peval_20261010]`.
B-noask (163) = the 93 unchanged Baseline B items (scores reused from the Baseline B P(eval) run) + the 70 regenerated no-ask items (new run).
Same scoring and bootstrap as metrics_b (item score = mean of samples; 1000 item bootstraps, percentile CI). Appends to <run_dir>/summary.md.
"""
import argparse

import numpy as np
import pandas as pd

from src.data_loaders import ROOT, load_config
from src.metrics import REPS, add_group, auc_boot, load_calls, md
from src.metrics_b import ORDER, fmt, strat_boot
from src.run_full import all_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--b-run", default="outputs/baselineB_peval_20261010")
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
    d1["grp"] = np.where(d1.group == "wildchat", "wildchat",
                         np.where(d1.group == "crisis", np.where(d1.turns == 1, "bench_single", "bench_multi"), "other"))
    s1 = d1[d1.grp != "other"].groupby(["model", "grp", "item_id"]).agg(score=("p_eval", "mean"), words=("words", "first")).reset_index()

    na = pd.read_json(ROOT / "data/synthetic/baseline_b_noask.jsonl", lines=True)
    orig = pd.read_json(ROOT / "data/synthetic/baseline_b.jsonl", lines=True)
    dN, n_att, n_fail = load_calls(run_dir)
    dO, _, _ = load_calls(ROOT / a.b_run)
    new_ids = set(na.loc[na.variant == "noask", "item_id"])
    assert set(dN.item_id) == new_ids and len(new_ids) == 70, "noask run must cover exactly the 70 regenerated items"
    sN = dN.groupby(["model", "item_id"]).p_eval.mean().rename("score").reset_index()
    sO = dO.groupby(["model", "item_id"]).p_eval.mean().rename("score").reset_index()
    sNA = pd.concat([sN, sO[~sO.item_id.isin(new_ids)]], ignore_index=True).merge(
        na[["item_id", "words", "category", "disclosure", "severity", "variant"]], on="item_id")
    assert sNA.groupby("model").item_id.nunique().eq(163).all()
    sNA["grp"] = "B_noask"
    sB = sO.merge(orig[["item_id", "words", "category", "disclosure", "severity"]], on="item_id")
    sB["grp"] = "B"
    cols = ["model", "grp", "item_id", "score", "words"]
    allsc = pd.concat([s1, sNA[cols], sB[cols]], ignore_index=True)

    wild_w = allsc[(allsc.model == "sonnet") & (allsc.grp == "wildchat")].words
    edges = np.quantile(np.log(wild_w), [.2, .4, .6, .8])
    comps = [("B_noask", "wildchat"), ("bench_single", "B_noask"), ("bench_multi", "B_noask"),
             ("B", "wildchat"), ("bench_single", "B")]  # last two = original Baseline B, for reference
    rows = []
    for model in ORDER:
        m = allsc[allsc.model == model]
        for p, n in comps:
            P, N = m[m.grp == p], m[m.grp == n]
            x, y = P.score.to_numpy(), N.score.to_numpy()
            raw = auc_boot(x, y, rng)
            lm = strat_boot(x, y, np.log(P.words.to_numpy()), np.log(N.words.to_numpy()), edges, rng)
            rows.append({"comparison": f"{p} ({len(x)}) vs {n} ({len(y)})", "model": model, "AUC [95% CI]": fmt(raw), "length-matched": fmt(lm)})
    auc_df = pd.DataFrame(rows)

    def piv(df, col):
        return df.pivot(index="comparison", columns="model", values=col)[ORDER].reindex(
            [f"{p} ({(allsc[(allsc.model=='sonnet')&(allsc.grp==p)]).shape[0]}) vs {n} ({(allsc[(allsc.model=='sonnet')&(allsc.grp==n)]).shape[0]})"
             for p, n in comps]).reset_index()

    # covert vs direct: AUC of covert (positive) vs direct, and mean P(eval) by disclosure, within B-noask and within B
    disc_rows, disc_means = [], []
    for name, d in (("B_noask", sNA), ("B", sB)):
        for model in ORDER:
            m = d[d.model == model]
            cov, dire = m[m.disclosure == "covert"], m[m.disclosure == "direct"]
            x, y = cov.score.to_numpy(), dire.score.to_numpy()
            raw = auc_boot(x, y, rng)
            lm = strat_boot(x, y, np.log(cov.words.to_numpy()), np.log(dire.words.to_numpy()), edges, rng)
            disc_rows.append({"set": name, "model": model, "AUC covert>direct [95% CI]": fmt(raw), "length-matched": fmt(lm),
                              "mean covert": f"{x.mean():.3f}", "mean direct": f"{y.mean():.3f}", "diff": f"{x.mean() - y.mean():+.3f}"})
    disc_df = pd.DataFrame(disc_rows)

    mean_group = allsc.groupby(["model", "grp"]).score.mean().unstack().reindex(ORDER)[["bench_single", "bench_multi", "B", "B_noask", "wildchat"]]
    cnt = allsc[allsc.model == "sonnet"].groupby("grp").item_id.nunique()
    mean_group.columns = [f"{c} (n={cnt[c]})" for c in mean_group.columns]

    # paired change on the 70 regenerated items: same id/category/disclosure/severity/length target, ask removed
    pair = sN.merge(sO, on=["model", "item_id"], suffixes=("_noask", "_B"))
    pair["delta"] = pair.score_noask - pair.score_B
    prow = []
    for model in ORDER:
        p = pair[pair.model == model]
        bs = [p.delta.to_numpy()[rng.integers(0, len(p), len(p))].mean() for _ in range(REPS)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        prow.append({"model": model, "mean B (70)": f"{p.score_B.mean():.3f}", "mean B-noask (70)": f"{p.score_noask.mean():.3f}",
                     "paired diff [95% CI]": f"{p.delta.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"})
    paired_df = pd.DataFrame(prow)

    ok = dN.groupby("model").size().reindex(ORDER)
    over = dN.groupby("model").reason_words.apply(lambda s: int((s > 15).sum())).reindex(ORDER)
    sd = dN.groupby(["model", "item_id"]).p_eval.std(ddof=1).groupby("model").mean().reindex(ORDER).round(3)

    L = ["", "## B-noask ablation", "",
         f"Question: is the Baseline B vs WildChat gap (and the lower-than-benchmark B scores) driven by the unrelated everyday ask in 70 of the 163 B items? "
         f"The 70 `everyday_ask` items were regenerated without the ask (same id, category, disclosure, severity, length target, style flags; "
         f"generation summary in `outputs/{sorted((ROOT / 'outputs').glob('baselineB_noask_2*'))[-1].name}/summary.md`). The other 93 items and their scores are reused unchanged from `{run_dir.parent.name}/{a.b_run.split('/')[-1]}`. "
         f"P(eval) was run only on the 70 new items: same prompt, models and 3 samples as Phase 1 and Baseline B (run `{run_dir.name}`), "
         f"{len(dN)} successful calls of 840 expected ({n_att} attempts, {n_fail} failed), cost ${dN.cost_usd.sum():.2f} (cap $3). "
         f"Soft 15-word reason cap overruns: " + ", ".join(f"{m} {over[m]}/{ok[m]}" for m in ORDER) + ". "
         "Per-item SD across samples: " + ", ".join(f"{m} {sd[m]}" for m in ORDER) + ".",
         "", "Caveats: only one regeneration per item (no repeat to measure regeneration noise); the 70 new items also differ from the originals in wording, "
         "not just the ask; the 93 retained items are those the original plan did not flag for an ask.", "",
         "### Mean P(eval) by group (mean of item scores)", "", md(mean_group.round(3).reset_index()), "",
         "### Paired change on the 70 regenerated items (B version vs no-ask version, same plan row)", "", md(paired_df), "",
         "### Test-likeness AUC, raw (first-named group is positive; 1000 item bootstraps, 95% percentile CI)", "", md(piv(auc_df, "AUC [95% CI]")), "",
         "### Test-likeness AUC, length-matched (WildChat length quintiles, log words)", "", md(piv(auc_df, "length-matched")), "",
         "Rows with `B` in place of `B_noask` are the original Baseline B numbers, repeated for comparison.", "",
         "### Covert vs direct disclosure within B-noask (and within original B for comparison)", "",
         "AUC with covert as the positive group: > 0.5 means covert items look more test-like than direct ones.", "", md(disc_df), ""]
    with open(run_dir / "summary.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))
    auc_df.to_csv(run_dir / "auc.csv", index=False)
    disc_df.to_csv(run_dir / "disclosure_auc.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
