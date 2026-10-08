"""Loaders for the two benchmark sources. Reads local files only; no network, no API calls."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["item_id", "source", "category", "label_origin", "n_turns", "text", "text_first_turn"]


def load_config(path=ROOT / "config.yaml"):
    return yaml.safe_load(Path(path).read_text())


def load_crisis_benchmark(directory=None) -> pd.DataFrame:
    """Between Help and Harm: validation (human labels) + test (gpt-4o-mini labels)."""
    d = ROOT / (directory or load_config()["data"]["crisis_benchmark_dir"])
    ann = d / "annotations"
    parts = []
    for split, label_file, origin in [
        ("validation", "human_merged_validation.parquet", "human"),
        ("test", "llm_merged_test_gpt-4o-mini.parquet", "gpt-4o-mini"),
    ]:
        ex = pd.read_parquet(d / f"{split}.parquet")
        lab = pd.read_parquet(ann / label_file)[["example_id", "final_label"]]
        m = ex.merge(lab, on="example_id", how="left", validate="1:1")
        # Tied votes have no resolved label; keep them, tagged "unresolved".
        m["final_label"] = m["final_label"].fillna("unresolved")
        parts.append(pd.DataFrame({
            "item_id": m["example_id"],
            "source": "crisis_benchmark",
            "category": m["final_label"],
            "label_origin": origin,
            "n_turns": m["n_turns"],
            "text": ["\n".join(t) for t in m["inputs"]],
            "text_first_turn": [t[0] for t in m["inputs"]],  # sensitivity variant
        }))
    return pd.concat(parts, ignore_index=True)[COLUMNS]


def load_counselbench_adv(csv_path=None) -> pd.DataFrame:
    """CounselBench-Adv: 6 columns (failure modes) x 20 expert-written questions."""
    p = ROOT / (csv_path or load_config()["data"]["counselbench_adv_csv"])
    wide = pd.read_csv(p)
    long = wide.melt(var_name="category", value_name="text").dropna(subset=["text"])
    long["item_id"] = [f"adv_{c}_{i:02d}" for c, i in
                       zip(long["category"], long.groupby("category").cumcount())]
    long["source"] = "counselbench_adv"
    long["label_origin"] = "expert"
    long["n_turns"] = 1
    long["text_first_turn"] = long["text"]
    return long.reset_index(drop=True)[COLUMNS]


def sample_crisis_benchmark(df: pd.DataFrame, seed: int, max_per_category=30, n_control=40) -> pd.DataFrame:
    """Phase 1 crisis sample: drop duplicate texts and unresolved labels, take up to
    max_per_category per crisis category (all of the small ones), plus n_control no_crisis."""
    df = df.drop_duplicates("text", keep="first")  # validation rows come first (human labels)
    df = df[df["category"] != "unresolved"]
    crisis = df[df["category"] != "no_crisis"]
    parts = [g.sample(n=min(len(g), max_per_category), random_state=seed)
             for _, g in crisis.groupby("category")]
    control = df[df["category"] == "no_crisis"].sample(n=n_control, random_state=seed)
    return pd.concat(parts + [control], ignore_index=True)


if __name__ == "__main__":
    cfg = load_config()
    sample = sample_crisis_benchmark(load_crisis_benchmark(), cfg["seed"])
    print(f"Phase 1 crisis sample: {len(sample)} items")
    print(sample.groupby(["category", "label_origin"]).size().unstack(fill_value=0))
    for name, df in [("crisis_benchmark", load_crisis_benchmark()),
                     ("counselbench_adv", load_counselbench_adv())]:
        print(f"\n=== {name}: {len(df)} items, {df['text'].duplicated().sum()} duplicate texts")
        print(df.groupby(["category", "label_origin"]).size().unstack(fill_value=0))
        print("n_turns:", df["n_turns"].describe()[["min", "50%", "max"]].to_dict())
        print("chars:  ", df["text"].str.len().describe()[["min", "50%", "max"]].to_dict())
