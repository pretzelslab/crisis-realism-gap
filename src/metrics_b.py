"""Baseline B analysis per guidelines v2: `uv run python -m src.metrics_b outputs/<baselineB_peval_run> [--phase1 outputs/full_20261008]`.
Item score = mean of the item's samples. AUC CIs: 1000 item bootstraps (percentile). Writes summary.md in the run dir.
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data_loaders import ROOT, load_config
from src.metrics import REPS, _strat_auc, add_group, auc_boot, load_calls, md
from src.run_full import all_items

ORDER = ["sonnet", "opus", "gpt", "gemini"]


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def strat_boot(x, y, wx, wy, edges, rng):
    est = _strat_auc(x, y, wx, wy, edges)
    boots = []
    for _ in range(REPS):
        i, j = rng.integers(0, len(x), len(x)), rng.integers(0, len(y), len(y))
        boots.append(_strat_auc(x[i], y[j], wx[i], wy[j], edges))
    return est, *np.percentile(boots, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--phase1", default="outputs/full_20261008")
    a = ap.parse_args()
    run_dir, cfg = ROOT / a.run_dir, load_config()
    rng = np.random.default_rng(cfg["seed"])

    # Phase 1 groups (existing calls, no new API calls)
    items = all_items(cfg)
    d1, _, _ = load_calls(ROOT / a.phase1)
    d1 = add_group(d1, items)
    meta = items.set_index("item_id")
    d1["turns"] = d1.item_id.map(meta.n_turns)
    d1["words"] = d1.item_id.map(meta.text.str.split().str.len())
    d1["grp"] = np.where(d1.group == "wildchat", "wildchat",
                         np.where(d1.group == "crisis", np.where(d1.turns == 1, "bench_single", "bench_multi"), "other"))
    s1 = d1[d1.grp != "other"].groupby(["model", "grp", "item_id"]).agg(score=("p_eval", "mean"), words=("words", "first")).reset_index()

    # Baseline B
    bb = pd.read_json(ROOT / "data/synthetic/baseline_b.jsonl", lines=True)
    dB, n_att, n_fail = load_calls(run_dir)
    sB = dB.groupby(["model", "item_id"]).p_eval.mean().rename("score").reset_index()
    sB = sB.merge(bb.drop(columns=["target_words"]), on="item_id")
    sB["grp"] = "B"
    sB["style_ask"] = sB["style"].map(lambda s: s["everyday_ask"])
    sB["has_app"] = sB.text.str.contains(r"\bapps?\b", case=False)
    allsc = pd.concat([s1, sB[["model", "grp", "item_id", "score", "words"]]], ignore_index=True)

    wild_w = allsc[(allsc.model == "sonnet") & (allsc.grp == "wildchat")].words
    edges = np.quantile(np.log(wild_w), [.2, .4, .6, .8])
    comps = [("bench_single", "B", "PRIMARY: bench single-turn crisis vs B (H1 expects >= 0.75)"),
             ("B", "wildchat", "PRIMARY: B vs WildChat (H2 expects <= 0.65)"),
             ("bench_single", "wildchat", "reference: bench single-turn vs WildChat"),
             ("bench_multi", "B", "extra: bench multi-turn crisis vs B")]
    rows, lm_rows, regs = [], [], []
    for model in ORDER:
        m = allsc[allsc.model == model]
        for p, n, label in comps:
            P, N = m[m.grp == p], m[m.grp == n]
            x, y = P.score.to_numpy(), N.score.to_numpy()
            raw = auc_boot(x, y, rng)
            lm = strat_boot(x, y, np.log(P.words.to_numpy()), np.log(N.words.to_numpy()), edges, rng)
            rows.append({"comparison": f"{p} ({len(x)}) vs {n} ({len(y)})", "model": model, "AUC [95% CI]": fmt(raw), "length-matched": fmt(lm)})
        for p, n, label in comps[:2]:
            z = m[m.grp.isin([p, n])]
            X = pd.DataFrame({"grp": (z.grp == p) * 1.0, "logw": np.log(z.words.to_numpy())}, index=z.index)
            fit = sm.GLM(z.score, sm.add_constant(X), family=sm.families.Binomial()).fit(cov_type="HC1")
            naive = sm.GLM(z.score, sm.add_constant(X[["grp"]]), family=sm.families.Binomial()).fit(cov_type="HC1")
            ci = fit.conf_int().loc["grp"]
            regs.append({"comparison": f"{p} vs {n}", "model": model,
                         "log-odds of {pos} (adj. log words)": f"{fit.params['grp']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]",
                         "p": f"{fit.pvalues['grp']:.3g}", "unadjusted": f"{naive.params['grp']:+.2f}",
                         "log-words coef": f"{fit.params['logw']:+.2f}"})
    auc_df = pd.DataFrame(rows)

    def piv(df, col):
        return df.pivot(index="comparison", columns="model", values=col)[ORDER].reset_index()

    # means
    sb = sB.copy()
    mean_group = allsc.groupby(["model", "grp"]).score.mean().unstack().reindex(ORDER)[["bench_single", "bench_multi", "B", "wildchat"]]
    cnt = allsc.groupby("grp").item_id.nunique()
    mean_group.columns = [f"{c} (n={cnt[c]})" for c in mean_group.columns]

    def by(col):
        t = sb.groupby(["model", col]).score.mean().unstack().reindex(ORDER)
        t["mean of models"] = t.mean(axis=1)
        return t.round(3).reset_index()

    sb["everyday_ask"] = sb["style_ask"].map({True: "ask", False: "no ask"})
    sb["app"] = sb["has_app"].map({True: "mentions app", False: "no app"})
    n_ask, n_app = int(sb.drop_duplicates("item_id").style_ask.sum()), int(sb.drop_duplicates("item_id").has_app.sum())
    sd = dB.groupby(["model", "item_id"]).p_eval.std(ddof=1).groupby("model").mean().reindex(ORDER).round(3)
    over = dB.groupby("model").reason_words.apply(lambda s: int((s > 15).sum())).reindex(ORDER)
    ok = dB.groupby("model").size().reindex(ORDER)

    L = ["", f"# Baseline B P(eval): {run_dir.name}", "",
         f"{len(dB)} successful calls (163 items x 3 samples x 4 models = 1956 expected); calls.jsonl holds {n_att} attempts, {n_fail} failed "
         f"(78 Gemini `AttributeError` from a thread race on the shared client, fixed in `classify_eval.py` by holding a local reference; 2 socket errors; "
         f"all retried on resume, 0 remaining). Cost ${dB.cost_usd.sum():.2f} for successful calls (cap $5). Same prompt and models as Phase 1; reasons stored (synthetic).",
         f"Soft 15-word reason cap overruns: " + ", ".join(f"{m} {over[m]}/{ok[m]}" for m in ORDER) + ".",
         f"Per-item SD across 3 samples (mean over items): " + ", ".join(f"{m} {sd[m]}" for m in ORDER) + ".", "",
         "## Mean P(eval) by group (mean of item scores)", "", md(mean_group.round(3).reset_index()), "",
         "## Test-likeness AUC (item score = mean of samples; 1000 item bootstraps, 95% percentile CI)", "",
         "Positive group is the first named; 0.5 = indistinguishable. `length-matched` = AUC within WildChat length quintiles (log words), pooled by pair count.", "",
         "### AUC, raw", "", md(piv(auc_df, "AUC [95% CI]")), "",
         "### AUC, length-matched", "", md(piv(auc_df, "length-matched")), "",
         "### Fractional logit, P(eval) ~ group + log(words), robust (HC1) SEs", "", md(pd.DataFrame(regs)), "",
         "## Mean P(eval) in Baseline B by category", "", md(by("category")), "",
         "## ... by disclosure", "", md(by("disclosure")), "",
         "## ... by severity", "", md(by("severity")), "",
         f"## ... by everyday-ask flag ({n_ask} flagged of 163) and by the word \"app\" ({n_app} items)", "",
         md(by("everyday_ask")), "", md(by("app")), "",
         "## Not computed (stated in spec, needs data not yet available)", "",
         "- bench crisis first turns (163) vs B: Phase 1 scored the full multi-turn text, not first turns alone; first-turn-only scores need new calls "
         "(for the 75 single-turn items the first turn is the whole text, so that part equals the primary comparison).",
         "- B (Haiku) vs B (hand-written): the 20 hand-written items are not written yet.", ""]
    with open(run_dir / "summary.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))
    auc_df.to_csv(run_dir / "auc.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
