"""Phase 1 memo figures from existing P(eval) calls (no API calls): `uv run python -m src.make_figures`.
Writes figures/{fig1_grid,fig2_auc,fig3_mean_peval}.png and figures/figure_data.csv (the plotted numbers).
Scores: item score = mean of 3 samples; AUC CIs = 1000 item bootstraps (percentile), same code as the analysis scripts.
Sets: benchmark crisis = all 163 crisis items (single + multi-turn); controls = 40 no_crisis items; B = Baseline B (163);
B no side question = 93 unchanged B items + 70 regenerated without the unrelated ask; C = Baseline C (163); WildChat = 200.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from src.data_loaders import ROOT, load_config
from src.metrics import add_group, auc_boot, load_calls
from src.run_full import all_items

MODELS = ["sonnet", "opus", "gpt", "gemini"]
MODEL_LABEL = {"sonnet": "Claude Sonnet 5.5", "opus": "Claude Opus 5.5", "gpt": "GPT 6.1 Sol", "gemini": "Gemini 3.8 Flash"}
# Okabe-Ito (colourblind-safe); each model also has its own marker so identity never rests on colour alone
COLOR = {"sonnet": "#0072B2", "opus": "#D55E00", "gpt": "#009E73", "gemini": "#CC79A7"}
MARK = {"sonnet": "o", "opus": "s", "gpt": "^", "gemini": "D"}
SET_LABEL = {"crisis": "Benchmark crisis items", "no_crisis": "Benchmark control items\n(no crisis)", "B": "Baseline B\n(synthetic crisis)",
             "B_noask": "Baseline B without\nside question", "C": "Baseline C\n(synthetic, no crisis)", "wildchat": "WildChat\n(real user messages)"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#dddddd"
P_DEF = "P(eval) is a model's estimated probability, from 0 to 1, that a message comes from a test rather than a real user (mean of 3 samples per item)."
CAPTION = ("AUC (area under the curve) is the chance that a randomly chosen item from the first set gets a higher P(eval) than a randomly chosen item from the second set: "
           "0.5 means the models cannot tell the sets apart, 1.0 means the first set always looks more like a test. " + P_DEF)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK})


def load_scores(cfg):
    items = all_items(cfg)
    meta = items.set_index("item_id")
    d1, _, _ = load_calls(ROOT / "outputs/full_20261008")
    d1 = add_group(d1, items)
    d1["words"] = d1.item_id.map(meta.text.str.split().str.len())
    s1 = d1[d1.group.isin(["crisis", "no_crisis", "wildchat"])].groupby(["model", "group", "item_id"]).p_eval.mean().rename("score").reset_index()
    s1 = s1.rename(columns={"group": "grp"})

    def run(path):
        d, _, _ = load_calls(ROOT / path)
        return d.groupby(["model", "item_id"]).p_eval.mean().rename("score").reset_index()

    sB, sN, sC = run("outputs/baselineB_peval_20261010"), run("outputs/baselineB_noask_peval_20261010"), run("outputs/baselineC_peval_20261010")
    sNA = pd.concat([sN, sB[~sB.item_id.isin(set(sN.item_id))]], ignore_index=True)
    out = pd.concat([s1, sB.assign(grp="B"), sNA.assign(grp="B_noask"), sC.assign(grp="C")], ignore_index=True)
    n = out[out.model == "sonnet"].groupby("grp").item_id.nunique().to_dict()
    assert n == {"B": 163, "B_noask": 163, "C": 163, "crisis": 163, "no_crisis": 40, "wildchat": 200}, n
    return out, n


def nice_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def finish(fig, name, caption=CAPTION):
    fig.text(0.01, 0.005, caption, ha="left", va="bottom", fontsize=8, color=MUTED, wrap=True)
    fig.savefig(ROOT / "figures" / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    cfg = load_config()
    (ROOT / "figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    sc, n = load_scores(cfg)

    comps = [("crisis", "wildchat"), ("no_crisis", "wildchat"), ("B", "wildchat"), ("B_noask", "wildchat"), ("C", "wildchat"), ("B_noask", "C")]
    rows = []
    for m in MODELS:
        for p, q in comps:
            x = sc[(sc.model == m) & (sc.grp == p)].score.to_numpy()
            y = sc[(sc.model == m) & (sc.grp == q)].score.to_numpy()
            a, lo, hi = auc_boot(x, y, rng)
            rows.append({"model": m, "first_set": p, "second_set": q, "auc": a, "ci_lo": lo, "ci_hi": hi, "n_first": len(x), "n_second": len(y)})
    auc = pd.DataFrame(rows)
    mean = sc.groupby(["model", "grp"]).score.mean().rename("mean_peval").reset_index()
    pd.concat([auc.assign(table="auc"), mean.assign(table="mean_peval")]).to_csv(ROOT / "figures/figure_data.csv", index=False)

    # ---------- Figure 1: 2x2 grid, AUC vs WildChat
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2), sharex=False)
    cells = {(0, 0): ("missing", None), (0, 1): ("auc", "B"), (1, 0): ("ref", "wildchat"), (1, 1): ("auc", "C")}
    titles = {(0, 0): "Real crisis messages", (0, 1): "Baseline B (synthetic crisis)", (1, 0): "WildChat (real, no crisis; reference)", (1, 1): "Baseline C (synthetic, no crisis)"}
    for (r, c), (kind, key) in cells.items():
        ax = axes[r, c]
        ax.set_title(titles[(r, c)], fontsize=10.5, loc="left", fontweight="bold")
        if kind == "auc":
            ax.set_xlim(0.2, 1.0)
            ax.set_ylim(0, 1)
            ax.axvline(0.5, color=MUTED, ls="--", lw=1)
            ax.set_yticks([])
            sub = auc[(auc.first_set == key) & (auc.second_set == "wildchat")].set_index("model")
            lo, hi = sub.auc.min(), sub.auc.max()
            ax.plot([lo, hi], [0.5, 0.5], color=INK, lw=6, alpha=0.18, solid_capstyle="round")
            for i, m in enumerate(MODELS):
                ax.scatter(sub.loc[m, "auc"], 0.5, s=70, color=COLOR[m], marker=MARK[m], edgecolor="white", linewidth=1, zorder=3)
            ax.text(0.97, 0.84, f"AUC {lo:.2f} to {hi:.2f} across the 4 models", ha="right", fontsize=10, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=3), zorder=6)
            ax.set_xlabel("AUC against WildChat (0.5 = indistinguishable)", fontsize=9)
            nice_axis(ax)
        elif kind == "ref":
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.58, "Reference set", ha="center", va="center", fontsize=12, color=INK, transform=ax.transAxes)
            ax.text(0.5, 0.38, f"{n['wildchat']} real user messages;\nAUC against itself is 0.5 by definition", ha="center", va="center", fontsize=9.5, color=MUTED, transform=ax.transAxes)
            for s in ax.spines.values():
                s.set_color(GRID)
        else:
            ax.set_xticks([]); ax.set_yticks([])
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f3f3f3", hatch="///", edgecolor=GRID, lw=0))
            ax.text(0.5, 0.5, "Missing\n(scrubbed from public data)", ha="center", va="center", fontsize=12, color=INK, transform=ax.transAxes,
                    bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.5"))
            for s in ax.spines.values():
                s.set_color(GRID)
    fig.text(0.04, 0.66, "Crisis", rotation=90, va="center", fontsize=12, fontweight="bold")
    fig.text(0.04, 0.34, "No crisis", rotation=90, va="center", fontsize=12, fontweight="bold")
    fig.text(0.28, 0.915, "Real", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.74, 0.915, "Synthetic", ha="center", fontsize=12, fontweight="bold")
    handles = [plt.Line2D([], [], marker=MARK[m], color=COLOR[m], ls="", markersize=8, label=MODEL_LABEL[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.075), fontsize=9)
    fig.suptitle("How easily models tell each kind of message from real user messages", fontsize=12.5, x=0.5, y=0.99)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.2, hspace=0.5, wspace=0.18)
    finish(fig, "fig1_grid.png")

    # ---------- Figure 2: AUC per comparison per model with CI bars
    labels = {("crisis", "wildchat"): "Benchmark crisis items\nvs WildChat", ("no_crisis", "wildchat"): "Benchmark control items\nvs WildChat",
              ("B", "wildchat"): "Baseline B\nvs WildChat", ("B_noask", "wildchat"): "Baseline B without side question\nvs WildChat",
              ("C", "wildchat"): "Baseline C\nvs WildChat", ("B_noask", "C"): "Baseline B without side question\nvs Baseline C"}
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    off = np.linspace(-0.27, 0.27, 4)
    for i, (p, q) in enumerate(comps):
        for j, m in enumerate(MODELS):
            r = auc[(auc.model == m) & (auc.first_set == p) & (auc.second_set == q)].iloc[0]
            y = i + off[j]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=COLOR[m], lw=2, solid_capstyle="round")
            ax.scatter(r.auc, y, s=46, color=COLOR[m], marker=MARK[m], edgecolor="white", linewidth=0.8, zorder=3)
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f6f6f6", zorder=0)
    ax.axvline(0.5, color=MUTED, ls="--", lw=1.2, zorder=1)
    ax.text(0.505, -0.62, "0.5 = models cannot tell the sets apart", fontsize=8.5, color=MUTED, va="center")
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels([labels[c] for c in comps], fontsize=9.5)
    ax.set_ylim(len(comps) - 0.5, -0.8)
    ax.set_xlim(0.2, 1.02)
    ax.set_xlabel("AUC (first-named set is the one expected to look more like a test); bars are 95% intervals")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    nice_axis(ax)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=4, fontsize=9)
    fig.suptitle("Which sets of messages do models flag as test-like?", fontsize=12.5, x=0.02, ha="left", fontweight="bold", y=1.0)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.2)
    finish(fig, "fig2_auc.png")

    # ---------- Figure 3: mean P(eval) by set, one line per model
    order = ["crisis", "no_crisis", "B", "B_noask", "C", "wildchat"]
    X3 = {"crisis": "Benchmark\ncrisis items", "no_crisis": "Benchmark control\nitems (no crisis)", "B": "Baseline B\n(synthetic crisis)",
          "B_noask": "Baseline B without\nside question", "C": "Baseline C\n(synthetic, no crisis)", "wildchat": "WildChat\n(real user messages)"}
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(order))
    NUDGE = {"sonnet": 0.0, "opus": 0.0, "gpt": 0.0, "gemini": -0.012}
    for m in MODELS:
        v = [mean[(mean.model == m) & (mean.grp == g)].mean_peval.iloc[0] for g in order]
        ax.plot(x, v, color=COLOR[m], marker=MARK[m], lw=2, ms=7, mec="white", mew=0.8)
        ax.text(x[-1] + 0.12, v[-1] + NUDGE[m], MODEL_LABEL[m], color=INK, va="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([X3[g] + chr(10) + f"n = {n[g]}" for g in order], fontsize=8.5)
    ax.set_ylim(0, 0.8)
    ax.set_xlim(-0.4, len(order) - 1 + 1.3)
    ax.set_ylabel("Mean P(eval)")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    nice_axis(ax)
    ax.set_title("How test-like each set looks, by model (higher = looks more like a test)", fontsize=12.5, loc="left", fontweight="bold")
    fig.text(0.01, 0.0, "", fontsize=1)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.2)
    finish(fig, "fig3_mean_peval.png", P_DEF)
    print(auc.round(3).to_string())
    print(mean.pivot(index="model", columns="grp", values="mean_peval").round(3).to_string())


if __name__ == "__main__":
    main()
