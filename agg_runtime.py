#!/usr/bin/env python3
"""Aggregate baseline-compatible per-fold runtime measurements."""

import os

import pandas as pd


NAME = {
    "integrated_gradients_base_abs": "IG",
    "gradientshap_abs": "GradSHAP",
    "timing": "TIMING",
    "magig": "MA-GIG",
    "iron": "IRON",
    "IRON": "IRON",
}

ORDER = ["IG", "GradSHAP", "TIMING", "MA-GIG", "IRON"]

INPUT_FILE = os.environ.get("RUNTIME_INPUT", "runtime_all.csv")
OUTPUT_FILE = os.environ.get("RUNTIME_TABLE_OUTPUT", "runtime_table.csv")


df = pd.read_csv(INPUT_FILE)
df["method"] = df["explainer"].map(NAME).fillna(df["explainer"])

# Baseline protocol: first compute one mean_ms per fold, then report the mean
# and pandas' sample standard deviation (ddof=1) across folds.
g = (
    df.groupby(["data", "method"])["mean_ms"]
    .agg(mean="mean", std="std", n_folds="count")
    .reset_index()
)

ig = g[g.method == "IG"].set_index("data")["mean"]


def relative_to_ig(row):
    if row["data"] not in ig.index:
        return float("nan")
    return row["mean"] / ig.loc[row["data"]]


g["rel_to_ig"] = g.apply(relative_to_ig, axis=1)
g["method"] = pd.Categorical(g["method"], ORDER, ordered=True)
g = g.sort_values(["data", "method"])

print(g.to_string(index=False))
print()

for dataset, subset in g.groupby("data", observed=False):
    print(f"--- {dataset} ---")
    for _, row in subset.dropna(subset=["method"]).iterrows():
        relative = (
            "--"
            if pd.isna(row["rel_to_ig"])
            else f"{row['rel_to_ig']:.2f}$\\times$"
        )
        print(
            f"{str(row['method']):10s} & "
            f"{row['mean']:.2f} $\\pm$ {row['std']:.2f} & "
            f"{relative} \\\\"
        )
    print()

table = g.dropna(subset=["method"]).copy()
table["method"] = table["method"].astype(str)
table["runtime_ms"] = table.apply(
    lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}",
    axis=1,
)
table["relative_to_ig"] = table["rel_to_ig"].apply(
    lambda value: "--" if pd.isna(value) else f"{value:.2f}×"
)
table = table[["data", "method", "runtime_ms", "relative_to_ig"]]
table.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(table.to_string(index=False))
print(f"\nSaved: {OUTPUT_FILE}")

if os.path.exists("vae_train_time.csv"):
    training = pd.read_csv("vae_train_time.csv")
    training = training.groupby("data")["train_sec"].mean()
    print("\n=== Auxiliary training (s, one-time per dataset-fold) ===")
    for dataset, seconds in training.items():
        print(f"{dataset:10s}  MA-GIG (VAE): {seconds:8.1f} s")
