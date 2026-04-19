import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────
FEATURES = [
    'age', 'gender', 'bmi',
    'smoking', 'alcohol_consumption', 'physical_activity', 'diet_quality',
    'chest_pain', 'shortness_of_breath', 'coughing',
    'avg_glucose_level', 'HbA1c_level',
    'cholesterol_total', 'blood_pressure',
    'serum_creatinine',
]

# ─────────────────────────────────────────────
# LOAD DATASETS (for sizing only)
# ─────────────────────────────────────────────
alz    = pd.read_csv("alzheimers_disease_data.csv")
brca   = pd.read_csv("brca.csv")
diab   = pd.read_csv("diabetic_data.csv")
stroke = pd.read_csv("healthcare-dataset-stroke-data.csv")
heart  = pd.read_csv("HeartDiseaseTrain-Test.csv")
kidney = pd.read_csv("kidney-stone-dataset.csv")
lung   = pd.read_csv("lung_disease_data.csv")

N = min(5000, len(alz), len(brca), len(diab), len(stroke),
        len(heart), len(kidney), len(lung))
print(f"\nUsing N={N} samples per disease")

# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────
def build_features(n, disease):

    data = pd.DataFrame({
        'age':                 np.random.randint(30, 80, n).astype(float),
        'gender':              np.random.randint(0, 2, n).astype(float),
        'bmi':                 np.random.normal(26, 4, n),
        'smoking':             np.random.binomial(1, 0.25, n).astype(float),
        'alcohol_consumption': np.random.randint(0, 3, n).astype(float),
        'physical_activity':   np.random.randint(1, 4, n).astype(float),
        'diet_quality':        np.random.randint(2, 6, n).astype(float),
        'chest_pain':          np.random.binomial(1, 0.15, n).astype(float),
        'shortness_of_breath': np.random.binomial(1, 0.15, n).astype(float),
        'coughing':            np.random.binomial(1, 0.12, n).astype(float),
        'avg_glucose_level':   np.random.normal(100, 15, n),
        'HbA1c_level':         np.random.normal(5.2, 0.5, n),
        'cholesterol_total':   np.random.normal(185, 22, n),
        'blood_pressure':      np.random.normal(115, 12, n),
        'serum_creatinine':    np.random.normal(0.9, 0.2, n),
    })

    data['disease'] = disease       #Adds a label column tagging every row with the disease name

    if disease == 'heart':
        data['chest_pain']           = np.random.binomial(1, 0.72, n).astype(float)
        data['shortness_of_breath']  = np.random.binomial(1, 0.60, n).astype(float)
        data['cholesterol_total']    += np.random.normal(55, 8, n)
        data['blood_pressure']       += np.random.normal(20, 6, n)
        data['age']                  += np.random.normal(8, 4, n)

    elif disease == 'diabetic':
        data['avg_glucose_level']    += np.random.normal(75, 12, n)
        data['HbA1c_level']          += np.random.normal(2.5, 0.5, n)
        data['bmi']                  += np.random.normal(5, 2, n)
        data['age']                  += np.random.normal(5, 4, n)
        data['blood_pressure']       += np.random.normal(5, 4, n)

    elif disease == 'stroke':
        data['blood_pressure']       += np.random.normal(45, 7, n)
        data['age']                  += np.random.normal(15, 5, n)
        data['avg_glucose_level']    += np.random.normal(12, 8, n)
        data['smoking']              = np.random.binomial(1, 0.65, n).astype(float)

    elif disease == 'kidney':
        data['serum_creatinine']     += np.random.normal(3.5, 0.5, n)
        data['blood_pressure']       += np.random.normal(12, 5, n)
        data['avg_glucose_level']    += np.random.normal(8, 6, n)
        data['age']                  += np.random.normal(5, 4, n)

    elif disease == 'lung':
        data['coughing']             = np.random.binomial(1, 0.80, n).astype(float)
        data['smoking']              = np.random.binomial(1, 0.75, n).astype(float)
        data['shortness_of_breath']  = np.random.binomial(1, 0.70, n).astype(float)
        data['chest_pain']           = np.random.binomial(1, 0.55, n).astype(float)
        data['age']                  += np.random.normal(5, 4, n)

    elif disease == 'alzheimers':
        data['age']                  += np.random.normal(20, 4, n)
        data['bmi']                  -= np.random.normal(4, 2, n)
        data['physical_activity']    -= np.random.normal(1.5, 0.5, n)
        data['diet_quality']         -= np.random.normal(1.5, 0.5, n)

    elif disease == 'brca':
        data['gender']               = np.ones(n)
        data['bmi']                  += np.random.normal(5, 2, n)
        data['age']                  += np.random.normal(5, 4, n)
        data['alcohol_consumption']  += np.random.normal(2.0, 0.5, n)

    for col, mu, sigma in [
        ('age',            2.0, 2.0),
        ('blood_pressure', 4.0, 4.0),
    ]:
        mask = np.random.rand(n) < 0.20             #creates a boolean array — True for ~20% of rows randomly
        data.loc[mask, col] += np.random.normal(mu, sigma, size=mask.sum())

    noise = np.random.normal(0, 4, data[FEATURES].shape)
    data[FEATURES] = data[FEATURES] + noise

    return data


# ─────────────────────────────────────────────
# BUILD BALANCED DATASET
# ─────────────────────────────────────────────
combined = pd.concat([
    build_features(N, 'heart'),
    build_features(N, 'diabetic'),
    build_features(N, 'stroke'),
    build_features(N, 'kidney'),
    build_features(N, 'lung'),
    build_features(N, 'alzheimers'),
    build_features(N, 'brca'),
], ignore_index=True)

combined = combined.dropna(subset=['disease'])

print("\nClass Distribution:")
print(combined['disease'].value_counts())

print("\nKey feature means per disease:")
print(combined.groupby('disease')[
    ['serum_creatinine', 'blood_pressure', 'avg_glucose_level',
     'cholesterol_total', 'coughing', 'smoking']
].mean().round(2))

# ─────────────────────────────────────────────
# SAVE META
# ─────────────────────────────────────────────
joblib.dump(combined[FEATURES].median().to_dict(), 'feature_medians.pkl')
joblib.dump(FEATURES, 'feature_cols.pkl')

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
MODEL_MAP = {
    'heart':      RandomForestClassifier(
                      n_estimators=100, max_depth=5,
                      min_samples_leaf=10, max_features='sqrt',
                      class_weight='balanced', random_state=42),

    'diabetic':   lgb.LGBMClassifier(
                      n_estimators=100, max_depth=5,
                      num_leaves=31, learning_rate=0.05,
                      min_child_samples=20,
                      class_weight='balanced',
                      verbose=-1,
                      random_state=42),

    'stroke':     GradientBoostingClassifier(
                      n_estimators=100, max_depth=4,
                      learning_rate=0.05,
                      min_samples_leaf=10, random_state=42),

    'kidney':     RandomForestClassifier(
                      n_estimators=100, max_depth=5,
                      min_samples_leaf=10, max_features='sqrt',
                      class_weight='balanced', random_state=42),

    'lung':       lgb.LGBMClassifier(
                      n_estimators=100, max_depth=5,
                      num_leaves=31, learning_rate=0.05,
                      min_child_samples=20,
                      class_weight='balanced',
                      verbose=-1,
                      random_state=42),

    'alzheimers': GradientBoostingClassifier(
                      n_estimators=100, max_depth=4,
                      learning_rate=0.05,
                      min_samples_leaf=10, random_state=42),

    'brca':       GradientBoostingClassifier(
                      n_estimators=100, max_depth=4,
                      learning_rate=0.05,
                      min_samples_leaf=10, random_state=42),
}

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
accuracies = {}

for disease in combined['disease'].unique():

    print(f"\n{'='*55}")
    print(f"  Training: {disease.upper()}")
    print(f"{'='*55}")

    X = combined[FEATURES]
    y = (combined['disease'] == disease).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURES
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURES
    )

    model = CalibratedClassifierCV(MODEL_MAP[disease], method='sigmoid', cv=3)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc    = accuracy_score(y_test, y_pred)
    accuracies[disease] = acc

    print(f"\n  Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=['No_' + disease, disease]))

    joblib.dump(model,  f"disease_model_{disease}.pkl")
    joblib.dump(scaler, f"scaler_{disease}.pkl")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  ALL MODELS TRAINED SUCCESSFULLY!")
print("="*55)
print(f"\n  {'Disease':<15} {'Accuracy':>12}")
print("  " + "-"*30)
for disease, acc in accuracies.items():
    print(f"  {disease:<15} {acc:>12.4f}")