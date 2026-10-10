"""B vs crisis-level WildChat AUC (topic-mismatch check for H2), no new P(eval) calls.
`uv run python -m src.wildchat_crisis_auc --labels outputs/<label_run> --b outputs/<baselineB_peval_run>`
Labels come from the label run's calls.jsonl (parsed rows only). If fewer than 200 items are labelled the section is
marked PROVISIONAL. Appends to <b>/summary.md. Only counts, ids-free tables and P(eval) numbers are written (no text).
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
    ap.add_argument("--labels", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--phase1", default="outputs/full_20261008")
    a = ap.parse_args()
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    lab = pd.read_json(ROOT / a.labels / "calls.jsonl", lines=True)
    lab = lab.dropna(subset=["crisis_level"]).drop_duplicates("item_id", keep="last")
    n_lab = len(lab)
    prov = n_lab < 200

    items = all_items(cfg)
    meta = items.set_index("item_id")
    d1, _, _ = load_calls(ROOT / a.phase1)
    d1 = add_group(d1, items)
    wc = d1[d1.group == "wildchat"].groupby(["model", "item_id"]).p_eval.mean().rename("score").reset_index()
    wc["words"] = wc.item_id.map(meta.text.str.split().str.len())
    wc = wc.merge(lab[["item_id", "crisis_level", "category", "confidence"]], on="item_id", how="inner")
    bb = pd.read_json(ROOT / "data/synthetic/baseline_b.jsonl", lines=True)
    dB, _, _ = load_calls(ROOT / a.b)
    sB = dB.groupby(["model", "item_id"]).p_eval.mean().rename("score").reset_index()
    sB["words"] = sB.item_id.map(bb.set_index("item_id").words)
    all_w = d1[(d1.group == "wildchat") & (d1.model == "sonnet")].drop_duplicates("item_id").item_id.map(meta.text.str.split().str.len())
    edges = np.quantile(np.log(all_w), [.2, .4, .6, .8])  # same WildChat quintile edges as metrics_b (all 200 items)

    yes = wc[(wc.crisis_level == "yes")]
    n_yes = yes.item_id.nunique()
    cats = yes.drop_duplicates("item_id").category.value_counts().to_dict()
    n_yes_hi = int(yes.drop_duplicates("item_id").confidence.ge(0.8).sum())
    rows = []
    for model in ORDER:
        b = sB[sB.model == model]
        for name, w in (("crisis-level WildChat", yes[yes.model == model]),
                        ("non-crisis WildChat (labelled)", wc[(wc.crisis_level == "no") & (wc.model == model)])):
            x, y = b.score.to_numpy(), w.score.to_numpy()
            raw = auc_boot(x, y, rng)
            lm = strat_boot(x, y, np.log(b.words.to_numpy()), np.log(w.words.to_numpy()), edges, rng)
            rows.append({"comparison": f"B ({len(x)}) vs {name} ({len(y)})", "model": model, "AUC [95% CI]": fmt(raw), "length-matched": fmt(lm)})
    auc = pd.DataFrame(rows)
    piv = lambda col: auc.pivot(index="comparison", columns="model", values=col)[ORDER].reset_index()
    # where do the crisis-level WildChat items sit in B's P(eval) distribution?
    pos = []
    for model in ORDER:
        b = sB[sB.model == model].score.to_numpy()
        w = yes[yes.model == model].score.to_numpy()
        pos.append({"model": model, "crisis-level WildChat scores": ", ".join(f"{v:.2f}" for v in sorted(w)),
                    "B median": f"{np.median(b):.2f}", "B min": f"{b.min():.2f}",
                    "share of B below the WildChat item (mean)": f"{np.mean([(b < v).mean() for v in w]):.2f}" if len(w) else "-"})
    tag = f"PROVISIONAL ({n_lab} of 200 WildChat items labelled)" if prov else "(all 200 WildChat items labelled)"
    L = ["", f"## Topic-mismatch check for H2: B vs crisis-level WildChat {tag}", "",
         f"Labeller: claude-haiku-4-5-20251001, prompt in `{a.labels}/prompt.txt`, six benchmark categories as the definition, any severity. "
         f"Labels from `{a.labels}/calls.jsonl`; no new P(eval) calls (Phase 1 WildChat scores reused).", "",
         f"Of {n_lab} labelled WildChat items: **{n_yes} crisis-level** ({n_yes_hi} with confidence >= 0.8), {int((lab.crisis_level == 'no').sum())} not. "
         f"Crisis-level by category: {cats}.", ""]
    if n_yes < 10:
        L += [f"**Caution: n = {n_yes} crisis-level items. The AUC below is dominated by those few items and its bootstrap CI is not reliable; "
              "treat it as an anecdote, not an estimate.**", ""]
    L += ["### AUC, raw (B is the positive group; 0.5 = indistinguishable)", "", md(piv("AUC [95% CI]")), "",
          "### AUC, length-matched (WildChat length quintiles, log words)", "", md(piv("length-matched")), "",
          "### Where the crisis-level WildChat items sit relative to B (item-mean P(eval))", "", md(pd.DataFrame(pos)), ""]
    with open(ROOT / a.b / "summary.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
