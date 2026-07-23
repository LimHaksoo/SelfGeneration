import os

import pandas as pd


NAME = {
    "integrated_gradients_base_abs": "IG",
    "gradientshap_abs": "GradSHAP",
    "timing": "TIMING",
    "magig": "MA-GIG",
    "iron": "IRON",
}
ORDER = ["IG", "GradSHAP", "TIMING", "MA-GIG", "IRON"]


df = pd.read_csv("runtime_all.csv")
df["method"] = df["explainer"].map(NAME).fillna(df["explainer"])

# One mean_ms value per fold, then sample std across the five fold means.
g = (
    df.groupby(["data", "method"])["mean_ms"]
    .agg(mean="mean", std="std", n_folds="count")
    .reset_index()
)

ig = g[g.method == "IG"].set_index("data")["mean"]
g["rel_to_ig"] = g.apply(
    lambda row: row["mean"] / ig[row["data"]],
    axis=1,
)

g["method"] = pd.Categorical(g["method"], ORDER, ordered=True)
g = g.sort_values(["data", "method"])

print(g.to_string(index=False))
print()

for data, subset in g.groupby("data", observed=False):
    print(f"--- {data} ---")
    for _, row in subset.iterrows():
        print(
            f"{row['method']:10s} & {row['mean']:.2f} "
            f"$\\pm$ {row['std']:.2f} & {row['rel_to_ig']:.2f}$\\times$ \\\\"
        )
    print()

table = g.copy()
table["runtime_ms"] = table.apply(
    lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}",
    axis=1,
)
table["relative_to_ig"] = table["rel_to_ig"].apply(
    lambda value: f"{value:.2f}×"
)
table = table[["data", "method", "runtime_ms", "relative_to_ig"]]
table.to_csv("runtime_table.csv", index=False, encoding="utf-8-sig")

print(table.to_string(index=False))
print("\nSaved: runtime_table.csv")

if os.path.exists("vae_train_time.csv"):
    train = pd.read_csv("vae_train_time.csv")
    train = train.groupby("data")["train_sec"].mean()
    print("\n=== Auxiliary training (s, one-time per dataset-fold) ===")
    for data, seconds in train.items():
        print(f"{data:10s}  MA-GIG (VAE): {seconds:8.1f} s")
