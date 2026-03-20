import pandas as pd
import os

# ── 1. File paths ────────────────────────────────────────────────────────────
files = {
    "alzheimers": "alzheimers_disease_data.csv",
    "brca":       "brca.csv",
    "diabetic":   "diabetic_data.csv",
    "stroke":     "healthcare-dataset-stroke-data.csv",
    "heart":      "HeartDiseaseTrain-Test.csv",
    "kidney":     "kidney-stone-dataset.csv",
    "lung":       "lung_disease_data.csv",
}

# ── 2. Load & tag each dataset ───────────────────────────────────────────────
dfs = []

for source_name, filename in files.items():
    df = pd.read_csv(filename)

    # Drop auto-generated unnamed index columns (e.g. "Unnamed: 0")
    df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], inplace=True)

    # Add a column to track which disease dataset each row came from
    df["disease_source"] = source_name

    print(f"[{source_name}]  rows: {len(df):>7,}  |  cols: {df.shape[1]}")
    dfs.append(df)

# ── 3. Combine all datasets (union of all columns) ───────────────────────────
#   sort=False  → preserves column order; missing columns filled with NaN
combined = pd.concat(dfs, ignore_index=True, sort=False)

# ── 4. Save ──────────────────────────────────────────────────────────────────
output_path = "Diseases_dataset.csv"
combined.to_csv(output_path, index=False)

print(f"\n Combined dataset saved '{output_path}'")
print(f"   Total rows : {len(combined):,}")
print(f"   Total cols : {combined.shape[1]}")