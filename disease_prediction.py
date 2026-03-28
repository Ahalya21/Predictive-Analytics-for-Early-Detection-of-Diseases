'''import pandas as pd
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
print(f"   Total cols : {combined.shape[1]}")'''










#preprocessing
'''import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Load the filled dataset ───────────────────────────────────────────────────
df = pd.read_csv('Diseases_dataset.csv', low_memory=False)

print("=== STEP 2: PREPROCESSING ===\n")
print(f"Input shape     : {df.shape}")
print(f"Missing values  : {df.isnull().sum().sum()}")

# ── Sub-step 1: Encode categorical columns → numbers ─────────────────────────
#
#  LabelEncoder converts text values to integers:
#    "Male" / "Female"  →  1 / 0
#    "Yes"  / "No"      →  1 / 0
#    "Steady" / "Up" / "Down"  →  0 / 1 / 2   (alphabetical order)
#
print("Encoding categorical columns...")

cat_cols = [c for c in df.select_dtypes(include='object').columns if c != 'disease_source']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # save encoder in case you need to reverse later

print(f"   Encoded : {len(cat_cols)} columns")
print(f"   Example : {cat_cols[:5]}")

# ── Sub-step 2: Scale numeric columns → mean=0, std=1 ────────────────────────
#
#  StandardScaler normalizes each numeric column so that:
#    - All features have equal weight in the ML model
#    - Large-range columns (e.g. cholesterol 100-300) don't dominate
#      small-range columns (e.g. smoking 0-1)
#
print(" Scaling numeric features...")

disease_col    = df['disease_source'].copy()          # preserve disease label
df_features    = df.drop(columns=['disease_source'])

num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
scaler   = StandardScaler()
df_features[num_cols] = scaler.fit_transform(df_features[num_cols])

df_features['disease_source'] = disease_col           # put disease column back
df = df_features.copy()

print(f"   Scaled  : {len(num_cols)} numeric columns")
print(f"   All features now have mean ≈ 0 and std ≈ 1")

# ── Verify & save ─────────────────────────────────────────────────────────────
print("=== RESULT ===")
print(f"   Shape         : {df.shape}")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Data types    : {df.dtypes.value_counts().to_dict()}")

output_file = 'unified_diseases_preprocessed.csv'
df.to_csv(output_file, index=False)

print(f"\n Preprocessed dataset saved → '{output_file}'")'''





























#feature engineering
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load preprocessed dataset
df = pd.read_csv('unified_diseases_preprocessed.csv', low_memory=False)

print("-------- FEATURE ENGINEERING --------")
print(f"Input  : {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"Disease counts:\n{df['disease_source'].value_counts()}\n")

# Separate features and label
disease_col = df['disease_source'].copy()
X_all       = df.drop(columns=['disease_source'])

# Balance dataset before ranking features
# Without this, diabetic (101k rows) dominates and only its
# columns get selected. Sample 500 rows per disease so all
# 7 diseases have equal influence on the ranking.
sample_size = 500
groups = []
for name, group in df.groupby('disease_source'):
    n = min(len(group), sample_size)
    groups.append(group.sample(n, random_state=42))
df_balanced = pd.concat(groups).reset_index(drop=True)

print(f"Balanced sample counts (used for ranking only):")
print(df_balanced['disease_source'].value_counts().to_string())

# Encode disease label -> integer for the selector model
le   = LabelEncoder()
Y = le.fit_transform(df_balanced['disease_source'].astype(str))
X = df_balanced.drop(columns=['disease_source'])

# Rank all features using Random Forest on balanced data
print("Ranking all features by importance...")
model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X, Y)

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("Top 20 most important features:")
print("-" * 50)
for i, (feat, score) in enumerate(importances.head(20).items(), 1):
    bar = '#' * int(score * 300)
    print(f"  {i:>2}. {feat:<35} {score:.4f}  {bar}")

# Select top 20 features
top_20 = importances.head(20).index.tolist()

# Apply selection to the FULL dataset (all 115,909 rows)
df_top            = X_all[top_20].copy()
df_top['disease_sources'] = disease_col.values

print(f"\nFinal dataset shape : {df_top.shape}")
print(f"  (from 152 columns down to {df_top.shape[1]})")
print(f"Missing values      : {df_top.isnull().sum().sum()}")
print(f"Disease counts:{df_top['disease_sources'].value_counts().to_string()}")

# Save outputs
df_top.to_csv('unified_diseases_features.csv', index=False)
joblib.dump(top_20, 'top_features.pkl')
joblib.dump(le,     'label_encoder.pkl')

print("Done!")
print("  Saved : unified_diseases_features.csv  (reduced dataset)")
print("  Saved : top_features.pkl (list of top 20 feature names)")
print("  Saved : label_encoder.pkl (disease name (integer mapping))")





import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the filled + preprocessed dataset (zero missing values)
df = pd.read_csv('unified_diseases_preprocessed.csv', low_memory=False)

print("-------- FEATURE ENGINEERING --------")
print(f"Input shape    : {df.shape}")
print(f"Missing values : {df.isnull().sum().sum()}")
print(f"Disease counts :\n{df['disease_source'].value_counts()}\n")

# Separate features and label
disease_col = df['disease_source'].copy()
X_all       = df.drop(columns=['disease_source'])

# Balance: sample 500 rows per disease so all diseases
# have equal influence on feature ranking.
# Without this, diabetic (101k rows) dominates and only
# its columns get selected.
sample_size = 500
groups = []
for name, group in df.groupby('disease_source'):
    n = min(len(group), sample_size)
    groups.append(group.sample(n, random_state=42))
df_balanced = pd.concat(groups).reset_index(drop=True)

print(f"Balanced sample (used for ranking only):")
print(df_balanced['disease_source'].value_counts().to_string())

# Encode disease label -> integer for the selector model
le = LabelEncoder()
Y  = le.fit_transform(df_balanced['disease_source'].astype(str))
X  = df_balanced.drop(columns=['disease_source'])

# Rank all features using Random Forest on balanced data
print("Ranking all features by importance...")
model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X, Y)

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop 20 most important features:")
print("-" * 50)
for i, (feat, score) in enumerate(importances.head(20).items(), 1):
    bar = '#' * int(score * 300)
    print(f"  {i:>2}. {feat:<35} {score:.4f}  {bar}")

# Select top 20 features
top_20 = importances.head(20).index.tolist()

# Apply selection to the FULL dataset (all 115,909 rows)
df_top           = X_all[top_20].copy()
df_top['disease_source'] = disease_col.values

print(f"\nFinal shape    : {df_top.shape}")
print(f"Missing values : {df_top.isnull().sum().sum()}")
print(f"\nDisease counts:\n{df_top['disease_source'].value_counts().to_string()}")

# Save outputs
df_top.to_csv('unified_diseases_dataset.csv', index=False)
joblib.dump(top_20, 'top_features.pkl')
joblib.dump(le,     'label_encoder.pkl')

print("Done!")
print("  Saved : unified_diseases_dataset.csv  (reduced dataset, no empty cells)")
print("  Saved : top_features.pkl               (list of top 20 feature names)")
print("  Saved : label_encoder.pkl              (disease name -> integer mapping)")'''


























# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.random.seed(42)

# =============================================================================
# STEP 1 -- Load and combine all 7 disease datasets
# =============================================================================
print("=" * 50)
print("STEP 1 -- LOAD DATA")
print("=" * 50)

files = {
    'alzheimers': 'alzheimers_disease_data.csv',
    'brca':       'brca.csv',
    'diabetic':   'diabetic_data.csv',
    'stroke':     'healthcare-dataset-stroke-data.csv',
    'heart':      'HeartDiseaseTrain-Test.csv',
    'kidney':     'kidney-stone-dataset.csv',
    'lung':       'lung_disease_data.csv',
}

dfs = []
for disease_name, filename in files.items():
    df_temp = pd.read_csv(filename)
    df_temp.drop(columns=[c for c in df_temp.columns if 'Unnamed' in str(c)],
                 inplace=True)
    df_temp['disease_source'] = disease_name
    print(f"  [{disease_name}]  rows: {len(df_temp):>7,}  cols: {df_temp.shape[1]}")
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True, sort=False)
print(f"\nCombined shape : {df.shape}")
print(f"Missing values : {df.isnull().sum().sum():,}")
print(f"\nDisease counts :\n{df['disease_source'].value_counts().to_string()}")

# =============================================================================
# STEP 2a -- Fill missing values PER disease group
#            (each disease fills from its own real values only)
# =============================================================================
print("\n" + "=" * 50)
print("STEP 2a -- FILL MISSING VALUES (per disease group)")
print("=" * 50)

filled = []
for disease, group in df.groupby('disease_source'):
    group = group.copy()
    for col in group.columns:
        if col == 'disease_source':
            continue
        missing_idx = group[col].isnull()
        if not missing_idx.any():
            continue
        real_vals = group[col].dropna()
        if len(real_vals) == 0:
            real_vals = df[col].dropna()   # fallback to global
        if len(real_vals) == 0:
            continue
        group.loc[missing_idx, col] = real_vals.sample(
            n=missing_idx.sum(), replace=True, random_state=42
        ).values
    filled.append(group)

df = pd.concat(filled).reset_index(drop=True)
print(f"Missing values after fill : {df.isnull().sum().sum()}")

# =============================================================================
# STEP 2b -- Encode categorical columns -> numbers
# =============================================================================
print("\n" + "=" * 50)
print("STEP 2b -- ENCODE CATEGORICAL COLUMNS")
print("=" * 50)

disease_col = df['disease_source'].copy()
df = df.drop(columns=['disease_source'])

for col in df.columns:
    if df[col].dtype == object or str(df[col].dtype).startswith('string'):
        le_col = LabelEncoder()
        df[col] = le_col.fit_transform(df[col].astype(str))

# Force all columns to numeric (safety check)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"All columns numeric : {df.select_dtypes(exclude=[np.number]).shape[1] == 0}")

# =============================================================================
# STEP 2c -- Scale numeric columns (mean~0, std~1)
# =============================================================================
print("\n" + "=" * 50)
print("STEP 2c -- SCALE NUMERIC COLUMNS")
print("=" * 50)

scaler     = StandardScaler()
df_scaled  = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['disease_source'] = disease_col.values

print(f"Shape after scaling : {df_scaled.shape}")
print(f"Missing values      : {df_scaled.isnull().sum().sum()}")

# Save intermediate preprocessed file
df_scaled.to_csv('unified_diseases_preprocessed.csv', index=False)
print("Saved -> unified_diseases_preprocessed.csv")

# =============================================================================
# STEP 3 -- Feature engineering
#           Balance by disease, rank features, keep top 20
# =============================================================================
print("\n" + "=" * 50)
print("STEP 3 -- FEATURE ENGINEERING")
print("=" * 50)

# Balance: sample 500 rows per disease so all diseases
# have equal influence on the feature ranking
sample_size = 500
groups_list = []
for name, group in df_scaled.groupby('disease_source'):
    n = min(len(group), sample_size)
    groups_list.append(group.sample(n, random_state=42))
df_balanced = pd.concat(groups_list).reset_index(drop=True)

print(f"Balanced sample counts (used for ranking only):")
print(df_balanced['disease_source'].value_counts().to_string())

le = LabelEncoder()
Y  = le.fit_transform(df_balanced['disease_source'].astype(str))
X  = df_balanced.drop(columns=['disease_source'])

print("\nRanking all features by importance...")
model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X, Y)

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop 20 most important features:")
print("-" * 50)
for i, (feat, score) in enumerate(importances.head(20).items(), 1):
    bar = '#' * int(score * 300)
    print(f"  {i:>2}. {feat:<35} {score:.4f}  {bar}")

top_20 = importances.head(20).index.tolist()

# Apply to FULL dataset
X_all  = df_scaled.drop(columns=['disease_source'])
df_top = X_all[top_20].copy()
df_top['disease_source'] = disease_col.values

print(f"\nFinal shape    : {df_top.shape}")
print(f"Missing values : {df_top.isnull().sum().sum()}")
print(f"\nDisease counts :\n{df_top['disease_source'].value_counts().to_string()}")
print("\nUnique values per column (all should be > 1):")
for col in df_top.columns:
    print(f"  {col:<35} unique = {df_top[col].nunique():>6}")

# Save final outputs
df_top.to_csv('unified_diseases_features.csv', index=False)
joblib.dump(top_20, 'top_features.pkl')
joblib.dump(le,     'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nDone!")
print("  Saved : unified_diseases_features.csv  (final dataset, no empty cells)")
print("  Saved : unified_diseases_preprocessed.csv")
print("  Saved : top_features.pkl")
print("  Saved : label_encoder.pkl")
print("  Saved : scaler.pkl")