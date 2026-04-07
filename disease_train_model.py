"""
disease_prediction.py  —  Final 7-disease pipeline
═══════════════════════════════════════════════════
FIXES in this version:
  FIX 1 — tumor_score: new combined feature (radius × concavity) = strong brca signal
  FIX 2 — brca_df zeroes out all shared features so model learns brca = tumor ONLY
  FIX 3 — predict.py has hard-rule override when tumor data is clearly present
"""
'''
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ═══════════════════════════════════════════════════════════
# COMPLETE FEATURE SET
# ═══════════════════════════════════════════════════════════
FEATURES = [
    'age', 'gender', 'bmi',
    'smoking', 'alcohol_consumption', 'physical_activity', 'diet_quality',
    'chest_pain', 'shortness_of_breath', 'coughing',
    'functional_assessment', 'adl_score',
    'tumor_radius', 'tumor_area', 'tumor_concavity',
    'tumor_score',                    # FIX 1 — new engineered feature
    'avg_glucose_level', 'HbA1c_level',
    'hypertension', 'heart_disease_history',
    'cholesterol_total', 'cholesterol_ldl', 'cholesterol_hdl', 'triglycerides',
    'max_heart_rate', 'chest_pain_type',
    'resting_bp', 'fasting_blood_sugar', 'exercise_angina', 'st_depression',
    'serum_creatinine', 'albumin', 'blood_urea',
    'blood_pressure', 'specific_gravity',
    'lung_capacity', 'yellow_fingers', 'anxiety',
    'ever_married', 'work_type',
]


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def gender_num(val):
    if pd.isna(val): return 0.0
    return 1.0 if str(val).strip().lower() in ('1', 'male', 'm') else 0.0

def to_num(s, default=0.0):
    return pd.to_numeric(s, errors='coerce').fillna(default)

def lcol(df, *names, default=0.0):
    col_map = {c.lower().strip().replace(' ', '_'): c for c in df.columns}
    for n in names:
        if n.lower().strip().replace(' ', '_') in col_map:
            return to_num(df[col_map[n.lower().strip().replace(' ', '_')]], default)
    return pd.Series([default] * len(df), dtype=float)

def smoke_encode(series, style='numeric'):
    s = series.astype(str).str.strip().str.lower()
    if style == 'history':
        m = {'never': 0, 'no info': 0, 'not current': 1,
             'former': 1, 'ever': 2, 'current': 3}
        return s.map(m).fillna(0)
    if style == 'status':
        m = {'never smoked': 0, 'unknown': 0,
             'formerly smoked': 1, 'smokes': 2}
        return s.map(m).fillna(0)
    return to_num(series, 0.0).clip(0, 3)

def build(disease_name, n, **kwargs):
    data = {}
    for f in FEATURES:
        val = kwargs.get(f, 0.0)
        data[f] = val.values if isinstance(val, pd.Series) else np.full(n, float(val))
    out = pd.DataFrame(data)
    out['disease_source'] = disease_name
    return out


# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ═══════════════════════════════════════════════════════════
print("\nSTEP 1 — LOAD DATASETS")

alz    = pd.read_csv("alzheimers_disease_data.csv")
brca   = pd.read_csv("brca.csv")
diab   = pd.read_csv("diabetic_data.csv")
stroke = pd.read_csv("healthcare-dataset-stroke-data.csv")
heart  = pd.read_csv("HeartDiseaseTrain-Test.csv")
kidney = pd.read_csv("kidney-stone-dataset.csv")
lung   = pd.read_csv("lung_disease_data.csv")

for name, df in [("alzheimers", alz), ("brca", brca), ("diabetic", diab),
                 ("stroke", stroke), ("heart", heart), ("kidney", kidney), ("lung", lung)]:
    print(f"  [{name:<12}]  rows: {len(df):>6}  cols: {df.shape[1]}")


# ═══════════════════════════════════════════════════════════
# STEP 1b — BALANCE DATASETS
# ═══════════════════════════════════════════════════════════
print("\nSTEP 1b — BALANCE DATASETS")

TARGET   = 4000
DIAB_CAP = 6000
NOISE    = 0.01


def augment_df(df, target_rows, noise_std=NOISE, random_state=42):
    rng = np.random.default_rng(random_state)
    if len(df) >= target_rows:
        return df
    n_needed = target_rows - len(df)
    sampled  = df.sample(n=n_needed, replace=True, random_state=random_state).copy()
    numeric_cols = [
        c for c in df.select_dtypes(include='number').columns
        if df[c].nunique() > 2 and df[c].std() > 0
    ]
    for col in numeric_cols:
        noise = rng.normal(0, noise_std * df[col].std(), size=len(sampled))
        sampled[col] = (sampled[col] + noise).clip(df[col].min(), df[col].max())
    return pd.concat([df, sampled], ignore_index=True)


def downsample_df(df, cap, random_state=42):
    if len(df) <= cap:
        return df
    return df.sample(n=cap, random_state=random_state).reset_index(drop=True)


print("  Before:")
for name, size in [("diabetic", len(diab)), ("lung", len(lung)), ("stroke", len(stroke)),
                   ("alzheimers", len(alz)), ("heart", len(heart)),
                   ("brca", len(brca)), ("kidney", len(kidney))]:
    print(f"    {name:<12}  {size:>7} rows")

diab   = downsample_df(diab,   cap=DIAB_CAP)
alz    = augment_df(alz,    target_rows=TARGET)
heart  = augment_df(heart,  target_rows=TARGET)
brca   = augment_df(brca,   target_rows=TARGET)
kidney = augment_df(kidney, target_rows=TARGET)

print("\n  After:")
for name, size in [("diabetic", len(diab)), ("lung", len(lung)), ("stroke", len(stroke)),
                   ("alzheimers", len(alz)), ("heart", len(heart)),
                   ("brca", len(brca)), ("kidney", len(kidney))]:
    print(f"    {name:<12}  {size:>7} rows")

total = len(diab)+len(lung)+len(stroke)+len(alz)+len(heart)+len(brca)+len(kidney)
print(f"\n  Total rows: {total:,}  (was ~116,000 before balancing)")


# ═══════════════════════════════════════════════════════════
# STEP 2 — MAP EACH DATASET → FEATURES
# ═══════════════════════════════════════════════════════════
print("\nSTEP 2 — MAP TO FEATURES")

# ── ALZHEIMERS ──────────────────────────────────────────────
alz_df = build('alzheimers', len(alz),
    age                   = lcol(alz, 'Age', 'age'),
    gender                = alz.get('Gender', pd.Series([0]*len(alz))).apply(gender_num),
    bmi                   = lcol(alz, 'BMI', 'bmi'),
    smoking               = smoke_encode(lcol(alz, 'Smoking', default=0)),
    alcohol_consumption   = lcol(alz, 'AlcoholConsumption'),
    physical_activity     = lcol(alz, 'PhysicalActivity'),
    diet_quality          = lcol(alz, 'DietQuality'),
    functional_assessment = lcol(alz, 'FunctionalAssessment'),
    adl_score             = lcol(alz, 'ADL'),
    cholesterol_total     = lcol(alz, 'CholesterolTotal'),
    cholesterol_ldl       = lcol(alz, 'CholesterolLDL'),
    cholesterol_hdl       = lcol(alz, 'CholesterolHDL'),
    triglycerides         = lcol(alz, 'CholesterolTriglycerides'),
    tumor_score           = pd.Series(np.zeros(len(alz))),   # no tumor → 0
)

# ── BREAST CANCER ───────────────────────────────────────────
# FIX 2: zero out ALL shared features so model learns
#         brca = tumor features ONLY, not age/bmi patterns
_radius    = lcol(brca, 'radius_mean')
_area      = lcol(brca, 'area_mean')
_concavity = lcol(brca, 'concavity_mean')
# FIX 1: tumor_score = radius × concavity — unique to cancer, very high signal
_tscore    = _radius * _concavity

brca_df = build('brca', len(brca),
    # Shared features → deliberately set to 0
    # This forces the model to rely ONLY on tumor features for brca
    age             = pd.Series(np.zeros(len(brca))),
    gender          = pd.Series(np.zeros(len(brca))),
    bmi             = pd.Series(np.zeros(len(brca))),
    smoking         = pd.Series(np.zeros(len(brca))),
    # Tumor features → real values
    tumor_radius    = _radius,
    tumor_area      = _area,
    tumor_concavity = _concavity,
    tumor_score     = _tscore,              # FIX 1
)

# ── DIABETIC ────────────────────────────────────────────────
_sh = diab.get('smoking_history', pd.Series(['never']*len(diab)))
diab_df = build('diabetic', len(diab),
    age                   = lcol(diab, 'age', 'Age'),
    gender                = diab.get('gender', pd.Series([0]*len(diab))).apply(gender_num),
    bmi                   = lcol(diab, 'bmi', 'BMI'),
    smoking               = smoke_encode(_sh, 'history'),
    avg_glucose_level     = lcol(diab, 'avg_glucose_level', 'blood_glucose_level'),
    HbA1c_level           = lcol(diab, 'HbA1c_level'),
    hypertension          = lcol(diab, 'hypertension', default=0),
    heart_disease_history = lcol(diab, 'heart_disease', default=0),
    tumor_score           = pd.Series(np.zeros(len(diab))),
)

# ── STROKE ──────────────────────────────────────────────────
_ss = stroke.get('smoking_status', pd.Series(['never smoked']*len(stroke)))
_em = stroke.get('ever_married',   pd.Series(['No']*len(stroke)))
_wt = stroke.get('work_type',      pd.Series(['Private']*len(stroke)))

stroke_df = build('stroke', len(stroke),
    age                   = lcol(stroke, 'age', 'Age'),
    gender                = stroke.get('gender', pd.Series([0]*len(stroke))).apply(gender_num),
    bmi                   = lcol(stroke, 'bmi', 'BMI'),
    smoking               = smoke_encode(_ss, 'status'),
    avg_glucose_level     = lcol(stroke, 'avg_glucose_level'),
    hypertension          = lcol(stroke, 'hypertension', default=0),
    heart_disease_history = lcol(stroke, 'heart_disease', default=0),
    ever_married          = _em.map({'Yes': 1, 'No': 0}).fillna(0),
    work_type             = _wt.map({'children': 0, 'Never_worked': 0,
                                     'Govt_job': 1, 'Self-employed': 2,
                                     'Private': 3}).fillna(0),
    tumor_score           = pd.Series(np.zeros(len(stroke))),
)

# ── HEART ───────────────────────────────────────────────────
_hg = heart.get('sex', heart.get('gender', heart.get('Gender',
        pd.Series([0]*len(heart)))))
heart_df = build('heart', len(heart),
    age                 = lcol(heart, 'age', 'Age'),
    gender              = _hg.apply(lambda x: 1.0 if str(x).strip()
                              in ('1','male','m','Male') else 0.0),
    bmi                 = lcol(heart, 'bmi', 'BMI'),
    cholesterol_total   = lcol(heart, 'cholestoral', 'chol', 'cholesterol'),
    max_heart_rate      = lcol(heart, 'thalach', 'Max_heart_rate', 'max_heart_rate'),
    chest_pain          = (lcol(heart, 'cp', 'chest_pain_type', default=0) > 0).astype(float),
    chest_pain_type     = lcol(heart, 'cp', 'chest_pain_type', default=0),
    resting_bp          = lcol(heart, 'trestbps', 'resting_bp'),
    blood_pressure      = lcol(heart, 'trestbps', 'resting_bp'),
    fasting_blood_sugar = lcol(heart, 'fbs', 'fasting_blood_sugar', default=0),
    exercise_angina     = lcol(heart, 'exang', 'exercise_angina', default=0),
    st_depression       = lcol(heart, 'oldpeak', 'st_depression', default=0),
    smoking             = lcol(heart, 'smoking', default=0),
    hypertension        = (lcol(heart, 'trestbps', 'resting_bp') > 140).astype(float),
    tumor_score         = pd.Series(np.zeros(len(heart))),
)

# ── KIDNEY ──────────────────────────────────────────────────
kidney_df = build('kidney', len(kidney),
    age               = lcol(kidney, 'age', 'Age'),
    bmi               = lcol(kidney, 'bmi', 'BMI'),
    blood_pressure    = lcol(kidney, 'bp', 'blood_pressure'),
    specific_gravity  = lcol(kidney, 'sg', 'specific_gravity'),
    albumin           = lcol(kidney, 'al', 'albumin'),
    avg_glucose_level = lcol(kidney, 'bgr', 'blood_glucose_random', 'avg_glucose_level'),
    blood_urea        = lcol(kidney, 'bu', 'blood_urea'),
    serum_creatinine  = lcol(kidney, 'sc', 'serum_creatinine'),
    hypertension      = (lcol(kidney, 'bp', 'blood_pressure') > 90).astype(float),
    tumor_score       = pd.Series(np.zeros(len(kidney))),
)

# ── LUNG ────────────────────────────────────────────────────
_lmap = {c.lower().strip().replace(' ', '_'): c for c in lung.columns}

def lung_col(*names, default=0.0):
    for n in names:
        k = n.lower().strip().replace(' ', '_')
        if k in _lmap:
            return to_num(lung[_lmap[k]], default)
    return pd.Series([default] * len(lung))

lung_df = build('lung', len(lung),
    age                 = lung_col('age'),
    gender              = lung.get(_lmap.get('gender', '__'),
                              pd.Series([0]*len(lung))).apply(gender_num),
    bmi                 = lung_col('bmi'),
    smoking             = smoke_encode(lung_col('smoking')),
    alcohol_consumption = lung_col('alcohol_consuming', 'alcohol_consumption'),
    physical_activity   = lung_col('physical_activity'),
    diet_quality        = lung_col('diet_quality'),
    chest_pain          = lung_col('chest_pain', 'chest pain'),
    shortness_of_breath = lung_col('shortness_of_breath', 'shortness of breath'),
    coughing            = lung_col('coughing'),
    lung_capacity       = lung_col('lung_capacity', 'lung capacity'),
    yellow_fingers      = lung_col('yellow_fingers', 'yellow fingers'),
    anxiety             = lung_col('anxiety'),
    hypertension        = lung_col('chronic_disease', 'chronic disease', default=0),
    tumor_score         = pd.Series(np.zeros(len(lung))),
)


# ═══════════════════════════════════════════════════════════
# STEP 3 — COMBINE
# ═══════════════════════════════════════════════════════════
print("\nSTEP 3 — COMBINE")

combined = pd.concat(
    [alz_df, brca_df, diab_df, stroke_df, heart_df, kidney_df, lung_df],
    ignore_index=True, sort=False
)
combined[FEATURES] = combined[FEATURES].fillna(0)

print(f"  Combined shape  : {combined.shape}")
print(f"  Total features  : {len(FEATURES)}")
print(f"  Missing values  : {combined[FEATURES].isnull().sum().sum()}")
print("  Disease counts  :")
print(combined['disease_source'].value_counts().to_string())


# ═══════════════════════════════════════════════════════════
# STEP 4 — ENCODE + SCALE
# ═══════════════════════════════════════════════════════════
print("\nSTEP 4 — ENCODE & SCALE")

le = LabelEncoder()
y  = le.fit_transform(combined['disease_source'])
X  = combined[FEATURES].values.astype(float)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  Classes : {list(le.classes_)}")
print(f"  X shape : {X_scaled.shape}")


# ═══════════════════════════════════════════════════════════
# STEP 5 — SPLIT
# ═══════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nSTEP 5 — SPLIT  →  Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")


# ═══════════════════════════════════════════════════════════
# STEP 6 — TRAIN  (LightGBM)
# ═══════════════════════════════════════════════════════════
print("\nSTEP 6 — TRAIN MODEL  (LightGBM)")

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

print(f"  Best iteration  : {model.best_iteration_}")
print("  Training complete.")


# ═══════════════════════════════════════════════════════════
# STEP 7 — EVALUATE
# ═══════════════════════════════════════════════════════════
print("\nSTEP 7 — EVALUATE")

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"  Accuracy : {acc * 100:.2f}%")

cm    = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print("\n  Confusion Matrix:")
print(cm_df.to_string())

importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n  Top 15 feature importances:")
for feat, imp in importances.head(15).items():
    bar = '█' * int(imp / importances.max() * 30)
    print(f"    {feat:<28} {imp:>6.0f}  {bar}")


# ═══════════════════════════════════════════════════════════
# STEP 8 — SAVE
# ═══════════════════════════════════════════════════════════
print("\nSTEP 8 — SAVE ARTIFACTS")

joblib.dump(model,    'disease_model.pkl')
joblib.dump(scaler,   'scaler.pkl')
joblib.dump(le,       'label_encoder.pkl')
joblib.dump(FEATURES, 'feature_cols.pkl')

print("  Saved : disease_model.pkl")
print("  Saved : scaler.pkl")
print("  Saved : label_encoder.pkl")
print("  Saved : feature_cols.pkl")
print("\nDone! Run predict.py to make predictions.")'''











































"""
disease_prediction.py  —  Final 7-disease pipeline
═══════════════════════════════════════════════════
FIXES applied:
  FIX 1 — tumor_score: combined feature (radius × concavity) = strong brca signal
  FIX 2 — brca_df uses realistic population values (NOT zeros) for shared features
  FIX 3 — stroke is now downsampled to TARGET (was never balanced before — main bias cause)
  FIX 4 — feature_medians.pkl saved so predict.py can use medians instead of zeros
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ═══════════════════════════════════════════════════════════
# COMPLETE FEATURE SET
# ═══════════════════════════════════════════════════════════
FEATURES = [
    'age', 'gender', 'bmi',
    'smoking', 'alcohol_consumption', 'physical_activity', 'diet_quality',
    'chest_pain', 'shortness_of_breath', 'coughing',
    'functional_assessment', 'adl_score',
    'tumor_radius', 'tumor_area', 'tumor_concavity',
    'tumor_score',
    'avg_glucose_level', 'HbA1c_level',
    'hypertension', 'heart_disease_history',
    'cholesterol_total', 'cholesterol_ldl', 'cholesterol_hdl', 'triglycerides',
    'max_heart_rate', 'chest_pain_type', 'heart_risk_score'
    'resting_bp', 'fasting_blood_sugar', 'exercise_angina', 'st_depression',
    'serum_creatinine', 'albumin', 'blood_urea',
    'blood_pressure', 'specific_gravity',
    'lung_capacity', 'yellow_fingers', 'anxiety',
    'ever_married', 'work_type',
]


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def gender_num(val):
    if pd.isna(val): return 0.0
    return 1.0 if str(val).strip().lower() in ('1', 'male', 'm') else 0.0

def to_num(s, default=0.0):
    return pd.to_numeric(s, errors='coerce').fillna(default)

def lcol(df, *names, default=0.0):
    col_map = {c.lower().strip().replace(' ', '_'): c for c in df.columns}
    for n in names:
        if n.lower().strip().replace(' ', '_') in col_map:
            return to_num(df[col_map[n.lower().strip().replace(' ', '_')]], default)
    return pd.Series([default] * len(df), dtype=float)

def smoke_encode(series, style='numeric'):
    s = series.astype(str).str.strip().str.lower()
    if style == 'history':
        m = {'never': 0, 'no info': 0, 'not current': 1,
             'former': 1, 'ever': 2, 'current': 3}
        return s.map(m).fillna(0)
    if style == 'status':
        m = {'never smoked': 0, 'unknown': 0,
             'formerly smoked': 1, 'smokes': 2}
        return s.map(m).fillna(0)
    return to_num(series, 0.0).clip(0, 3)

def build(disease_name, n, **kwargs):
    data = {}
    for f in FEATURES:
        val = kwargs.get(f, 0.0)
        data[f] = val.values if isinstance(val, pd.Series) else np.full(n, float(val))
    out = pd.DataFrame(data)
    out['disease_source'] = disease_name
    return out


# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ═══════════════════════════════════════════════════════════
print("\nSTEP 1 — LOAD DATASETS")

alz    = pd.read_csv("alzheimers_disease_data.csv")
brca   = pd.read_csv("brca.csv")
diab   = pd.read_csv("diabetic_data.csv")
stroke = pd.read_csv("healthcare-dataset-stroke-data.csv")
heart  = pd.read_csv("HeartDiseaseTrain-Test.csv")
kidney = pd.read_csv("kidney-stone-dataset.csv")
lung   = pd.read_csv("lung_disease_data.csv")


for name, df in [("alzheimers", alz), ("brca", brca), ("diabetic", diab),
                 ("stroke", stroke), ("heart", heart), ("kidney", kidney), ("lung", lung)]:
    print(f"  [{name:<12}]  rows: {len(df):>6}  cols: {df.shape[1]}")


# ═══════════════════════════════════════════════════════════
# STEP 1b — BALANCE DATASETS
# ═══════════════════════════════════════════════════════════
print("\nSTEP 1b — BALANCE DATASETS")

TARGET   = 4000
DIAB_CAP = 6000
NOISE    = 0.01


def augment_df(df, target_rows, noise_std=NOISE, random_state=42):
    rng = np.random.default_rng(random_state)
    if len(df) >= target_rows:
        return df
    n_needed = target_rows - len(df)
    sampled  = df.sample(n=n_needed, replace=True, random_state=random_state).copy()
    numeric_cols = [
        c for c in df.select_dtypes(include='number').columns
        if df[c].nunique() > 2 and df[c].std() > 0
    ]
    for col in numeric_cols:
        noise = rng.normal(0, noise_std * df[col].std(), size=len(sampled))
        sampled[col] = (sampled[col] + noise).clip(df[col].min(), df[col].max())
    return pd.concat([df, sampled], ignore_index=True)


def downsample_df(df, cap, random_state=42):
    if len(df) <= cap:
        return df
    return df.sample(n=cap, random_state=random_state).reset_index(drop=True)


print("  Before:")
for name, size in [("diabetic", len(diab)), ("lung", len(lung)), ("stroke", len(stroke)),
                   ("alzheimers", len(alz)), ("heart", len(heart)),
                   ("brca", len(brca)), ("kidney", len(kidney))]:
    print(f"    {name:<12}  {size:>7} rows")

diab   = downsample_df(diab,   cap=DIAB_CAP)
alz    = augment_df(alz,    target_rows=TARGET)
heart  = augment_df(heart,  target_rows=TARGET)
brca   = augment_df(brca,   target_rows=TARGET)
kidney = augment_df(kidney, target_rows=TARGET)

# ── FIX 3: stroke was never balanced — this was the PRIMARY cause of stroke bias ──
stroke = downsample_df(stroke, cap=TARGET)

print("\n  After:")
for name, size in [("diabetic", len(diab)), ("lung", len(lung)), ("stroke", len(stroke)),
                   ("alzheimers", len(alz)), ("heart", len(heart)),
                   ("brca", len(brca)), ("kidney", len(kidney))]:
    print(f"    {name:<12}  {size:>7} rows")

total = len(diab)+len(lung)+len(stroke)+len(alz)+len(heart)+len(brca)+len(kidney)
print(f"\n  Total rows: {total:,}")


# ═══════════════════════════════════════════════════════════
# STEP 2 — MAP EACH DATASET → FEATURES
# ═══════════════════════════════════════════════════════════
print("\nSTEP 2 — MAP TO FEATURES")

# ── ALZHEIMERS ──────────────────────────────────────────────
alz_df = build('alzheimers', len(alz),
    age                   = lcol(alz, 'Age', 'age'),
    gender                = alz.get('Gender', pd.Series([0]*len(alz))).apply(gender_num),
    bmi                   = lcol(alz, 'BMI', 'bmi'),
    smoking               = smoke_encode(lcol(alz, 'Smoking', default=0)),
    alcohol_consumption   = lcol(alz, 'AlcoholConsumption'),
    physical_activity     = lcol(alz, 'PhysicalActivity'),
    diet_quality          = lcol(alz, 'DietQuality'),
    functional_assessment = lcol(alz, 'FunctionalAssessment'),
    adl_score             = lcol(alz, 'ADL'),
    cholesterol_total     = lcol(alz, 'CholesterolTotal'),
    cholesterol_ldl       = lcol(alz, 'CholesterolLDL'),
    cholesterol_hdl       = lcol(alz, 'CholesterolHDL'),
    triglycerides         = lcol(alz, 'CholesterolTriglycerides'),
    tumor_score           = pd.Series(np.zeros(len(alz))),
)

# ── BREAST CANCER ───────────────────────────────────────────
# FIX 2: Use realistic population values instead of zeros for shared features.
# Original code zeroed age/bmi/etc which taught the model "brca = missing data person"
# causing real patients to never match brca. Now we use realistic distributions.
_radius    = lcol(brca, 'radius_mean')
_area      = lcol(brca, 'area_mean')
_concavity = lcol(brca, 'concavity_mean')
_tscore    = (_radius * _concavity)*0.3   # FIX 1: tumor_score = strong unique brca signal

np.random.seed(42)
brca_df = build('brca', len(brca),
    # Realistic population values — NOT zeros
    age             = pd.Series(np.random.normal(55, 12, len(brca)).clip(25, 90)),
    gender          = pd.Series(np.zeros(len(brca))),      # 0 = female, correct for brca
    bmi             = pd.Series(np.random.normal(27, 5, len(brca)).clip(18, 45)),
    smoking         = pd.Series(np.random.choice([0, 1, 2], len(brca), p=[0.6, 0.3, 0.1])),
    alcohol_consumption = pd.Series(np.random.normal(3, 2, len(brca)).clip(0, 14)),
    physical_activity   = pd.Series(np.random.normal(3, 2, len(brca)).clip(0, 10)),
    diet_quality        = pd.Series(np.random.normal(5, 2, len(brca)).clip(0, 10)),
    # Tumor features — real values (these are the key brca signals)
    tumor_radius    = _radius,
    tumor_area      = _area,
    tumor_concavity = _concavity,
    tumor_score     = _tscore,
)

# ── DIABETIC ────────────────────────────────────────────────
_sh = diab.get('smoking_history', pd.Series(['never']*len(diab)))
diab_df = build('diabetic', len(diab),
    age                   = lcol(diab, 'age', 'Age'),
    gender                = diab.get('gender', pd.Series([0]*len(diab))).apply(gender_num),
    bmi                   = lcol(diab, 'bmi', 'BMI'),
    smoking               = smoke_encode(_sh, 'history'),
    avg_glucose_level     = lcol(diab, 'avg_glucose_level', 'blood_glucose_level'),
    HbA1c_level           = lcol(diab, 'HbA1c_level'),
    hypertension          = lcol(diab, 'hypertension', default=0),
    heart_disease_history = lcol(diab, 'heart_disease', default=0),
    tumor_score           = pd.Series(np.zeros(len(diab))),
)

# ── STROKE ──────────────────────────────────────────────────
_ss = stroke.get('smoking_status', pd.Series(['never smoked']*len(stroke)))
_em = stroke.get('ever_married',   pd.Series(['No']*len(stroke)))
_wt = stroke.get('work_type',      pd.Series(['Private']*len(stroke)))

stroke_df = build('stroke', len(stroke),
    age                   = lcol(stroke, 'age', 'Age'),
    gender                = stroke.get('gender', pd.Series([0]*len(stroke))).apply(gender_num),
    bmi                   = lcol(stroke, 'bmi', 'BMI'),
    smoking               = smoke_encode(_ss, 'status'),
    avg_glucose_level     = lcol(stroke, 'avg_glucose_level'),
    hypertension          = lcol(stroke, 'hypertension', default=0),
    heart_disease_history = lcol(stroke, 'heart_disease', default=0),
    ever_married          = _em.map({'Yes': 1, 'No': 0}).fillna(0),
    work_type             = _wt.map({'children': 0, 'Never_worked': 0,
                                     'Govt_job': 1, 'Self-employed': 2,
                                     'Private': 3}).fillna(0),
    tumor_score           = pd.Series(np.zeros(len(stroke))),
)

# ── HEART ───────────────────────────────────────────────────
_hg = heart.get('sex', heart.get('gender', heart.get('Gender',
        pd.Series([0]*len(heart)))))
heart_df = build('heart', len(heart),
    age                 = lcol(heart, 'age', 'Age'),
    gender              = _hg.apply(lambda x: 1.0 if str(x).strip()
                              in ('1','male','m','Male') else 0.0),
    bmi                 = lcol(heart, 'bmi', 'BMI'),
    cholesterol_total   = lcol(heart, 'cholestoral', 'chol', 'cholesterol'),
    max_heart_rate      = lcol(heart, 'thalach', 'Max_heart_rate', 'max_heart_rate'),
    chest_pain          = (lcol(heart, 'cp', 'chest_pain_type', default=0) > 0).astype(float),
    chest_pain_type     = lcol(heart, 'cp', 'chest_pain_type', default=0),
    resting_bp          = lcol(heart, 'trestbps', 'resting_bp'),
    blood_pressure      = lcol(heart, 'trestbps', 'resting_bp'),
    fasting_blood_sugar = lcol(heart, 'fbs', 'fasting_blood_sugar', default=0),
    exercise_angina     = lcol(heart, 'exang', 'exercise_angina', default=0),
    st_depression       = lcol(heart, 'oldpeak', 'st_depression', default=0),
    smoking             = lcol(heart, 'smoking', default=0),
    hypertension        = (lcol(heart, 'trestbps', 'resting_bp') > 140).astype(float),
    tumor_score         = pd.Series(np.zeros(len(heart))),
    heart_risk_score = (
        lcol(heart, 'cholestoral') * 0.01 +
        lcol(heart, 'trestbps') * 0.02 +
        lcol(heart, 'oldpeak') * 2 +
        lcol(heart, 'thalach') * -0.01
    ),
)

# ── KIDNEY ──────────────────────────────────────────────────
kidney_df = build('kidney', len(kidney),
    age               = lcol(kidney, 'age', 'Age'),
    bmi               = lcol(kidney, 'bmi', 'BMI'),
    blood_pressure    = lcol(kidney, 'bp', 'blood_pressure'),
    specific_gravity  = lcol(kidney, 'sg', 'specific_gravity'),
    albumin           = lcol(kidney, 'al', 'albumin'),
    avg_glucose_level = lcol(kidney, 'bgr', 'blood_glucose_random', 'avg_glucose_level'),
    blood_urea        = lcol(kidney, 'bu', 'blood_urea'),
    serum_creatinine  = lcol(kidney, 'sc', 'serum_creatinine'),
    hypertension      = (lcol(kidney, 'bp', 'blood_pressure') > 90).astype(float),
    tumor_score       = pd.Series(np.zeros(len(kidney))),
)

# ── LUNG ────────────────────────────────────────────────────
_lmap = {c.lower().strip().replace(' ', '_'): c for c in lung.columns}

def lung_col(*names, default=0.0):
    for n in names:
        k = n.lower().strip().replace(' ', '_')
        if k in _lmap:
            return to_num(lung[_lmap[k]], default)
    return pd.Series([default] * len(lung))

lung_df = build('lung', len(lung),
    age                 = lung_col('age'),
    gender              = lung.get(_lmap.get('gender', '__'),
                              pd.Series([0]*len(lung))).apply(gender_num),
    bmi                 = lung_col('bmi'),
    smoking             = smoke_encode(lung_col('smoking')),
    alcohol_consumption = lung_col('alcohol_consuming', 'alcohol_consumption'),
    physical_activity   = lung_col('physical_activity'),
    diet_quality        = lung_col('diet_quality'),
    chest_pain          = lung_col('chest_pain', 'chest pain'),
    shortness_of_breath = lung_col('shortness_of_breath', 'shortness of breath'),
    coughing            = lung_col('coughing'),
    lung_capacity       = lung_col('lung_capacity', 'lung capacity'),
    yellow_fingers      = lung_col('yellow_fingers', 'yellow fingers'),
    anxiety             = lung_col('anxiety'),
    hypertension        = lung_col('chronic_disease', 'chronic disease', default=0),
    tumor_score         = pd.Series(np.zeros(len(lung))),
)


# ═══════════════════════════════════════════════════════════
# STEP 3 — COMBINE
# ═══════════════════════════════════════════════════════════
print("\nSTEP 3 — COMBINE")

combined = pd.concat(
    [alz_df, brca_df, diab_df, stroke_df, heart_df, kidney_df, lung_df],
    ignore_index=True, sort=False
)
combined[FEATURES] = combined[FEATURES].fillna(0)

print(f"  Combined shape  : {combined.shape}")
print(f"  Total features  : {len(FEATURES)}")
print(f"  Missing values  : {combined[FEATURES].isnull().sum().sum()}")
print("  Disease counts  :")
print(combined['disease_source'].value_counts().to_string())

# ── FIX 4: Save feature medians so predict.py can use them instead of zeros ──
feature_medians = combined[FEATURES].median().to_dict()

# Force disease-specific fields to 0 — median of combined data pulls these
# non-zero because of disease rows, which would inject tumor/lab values
# into general patients and bias predictions toward brca/heart/kidney etc.
FORCE_ZERO = [
    # brca
    'tumor_radius', 'tumor_area', 'tumor_concavity', 'tumor_score',
    # heart
    'chest_pain_type', 'exercise_angina', 'st_depression',
    'fasting_blood_sugar', 'max_heart_rate', 'resting_bp',
    # kidney
    'specific_gravity', 'albumin', 'blood_urea',
    # lung
    'lung_capacity', 'yellow_fingers', 'anxiety',
    # alzheimers
    'functional_assessment', 'adl_score',
]
for field in FORCE_ZERO:
    feature_medians[field] = 0.0

joblib.dump(feature_medians, 'feature_medians.pkl')
print("\n  Saved : feature_medians.pkl  (used by predict.py for missing field imputation)")


# ═══════════════════════════════════════════════════════════
# STEP 4 — ENCODE + SCALE
# ═══════════════════════════════════════════════════════════
print("\nSTEP 4 — ENCODE & SCALE")

le = LabelEncoder()
y  = le.fit_transform(combined['disease_source'])
X  = combined[FEATURES].values.astype(float)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  Classes : {list(le.classes_)}")
print(f"  X shape : {X_scaled.shape}")


# ═══════════════════════════════════════════════════════════
# STEP 5 — SPLIT
# ═══════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nSTEP 5 — SPLIT  →  Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")


# ═══════════════════════════════════════════════════════════
# STEP 6 — TRAIN  (LightGBM)
# ═══════════════════════════════════════════════════════════
print("\nSTEP 6 — TRAIN MODEL  (LightGBM)")

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

print(f"  Best iteration  : {model.best_iteration_}")
print("  Training complete.")


# ═══════════════════════════════════════════════════════════
# STEP 7 — EVALUATE
# ═══════════════════════════════════════════════════════════
print("\nSTEP 7 — EVALUATE")

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"  Accuracy : {acc * 100:.2f}%")

cm    = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print("\n  Confusion Matrix:")
print(cm_df.to_string())

importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n  Top 15 feature importances:")
for feat, imp in importances.head(15).items():
    bar = '█' * int(imp / importances.max() * 30)
    print(f"    {feat:<28} {imp:>6.0f}  {bar}")


# ═══════════════════════════════════════════════════════════
# STEP 8 — SAVE
# ═══════════════════════════════════════════════════════════
print("\nSTEP 8 — SAVE ARTIFACTS")

joblib.dump(model,    'disease_model.pkl')
joblib.dump(scaler,   'scaler.pkl')
joblib.dump(le,       'label_encoder.pkl')
joblib.dump(FEATURES, 'feature_cols.pkl')

print("  Saved : disease_model.pkl")
print("  Saved : scaler.pkl")
print("  Saved : label_encoder.pkl")
print("  Saved : feature_cols.pkl")
print("  Saved : feature_medians.pkl")
print("\nDone! Run predict.py to make predictions.")