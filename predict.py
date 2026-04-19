'''import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
feature_cols = joblib.load('feature_cols.pkl')
feature_medians = joblib.load('feature_medians.pkl')
le = joblib.load('label_encoder.pkl')

LABELS = {
    'alzheimers': "Alzheimer's disease",
    'brca': "Breast cancer",
    'diabetic': "Diabetes",
    'heart': "Heart disease",
    'kidney': "Kidney disease",
    'lung': "Lung cancer",
    'stroke': "Stroke",
}

# LOAD ALL MODELS
models = {}
scalers = {}

for d in LABELS.keys():
    models[d] = joblib.load(f"disease_model_{d}.pkl")
    scalers[d] = joblib.load(f"scaler_{d}.pkl")

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
print("\nEnter patient details (press Enter to skip):\n")

patient = {}

for f in feature_cols:
    val = input(f"{f}: ").strip()
    if val == "":
        patient[f] = feature_medians.get(f, 0)
    else:
        patient[f] = float(val)

# 👉 Important fix (avoid BRCA bias)
if patient.get('serum_creatinine', 0) == 0:
    patient['serum_creatinine'] = feature_medians.get('serum_creatinine', 1)

input_df = pd.DataFrame([patient])

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
disease_probs = {}

for d in LABELS.keys():
    scaled = scalers[d].transform(input_df)
    prob = models[d].predict_proba(scaled)[0][1]
    disease_probs[d] = prob

# BEST
disease = max(disease_probs, key=disease_probs.get)
confidence = disease_probs[disease] * 100

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
print("\nRESULT")
print("Predicted Disease:", LABELS[disease])
print(f"Confidence: {confidence:.2f}%\n")

print("All probabilities:\n")

for d, p in sorted(disease_probs.items(), key=lambda x: x[1], reverse=True):
    print(f"{LABELS[d]:<20} {p*100:.2f}%")'''




















import pandas as pd
import joblib

# LOAD
feature_cols = joblib.load('feature_cols.pkl')
feature_medians = joblib.load('feature_medians.pkl')

LABELS = {
    'alzheimers': "Alzheimer's Disease",
    'brca': "Breast Cancer",
    'diabetic': "Diabetes",
    'heart': "Heart Disease",
    'kidney': "Kidney Disease",
    'lung': "Lung Cancer",
    'stroke': "Stroke"
}

models = {}
scalers = {}

for d in LABELS:
    models[d] = joblib.load(f"disease_model_{d}.pkl")
    scalers[d] = joblib.load(f"scaler_{d}.pkl")


# INPUT
print("\nEnter Patient Details:\n")

patient = {}

for f in feature_cols:
    val = input(f"{f}: ").strip()

    if val == "":
        patient[f] = feature_medians[f]
    else:
        patient[f] = float(val)

input_df = pd.DataFrame([patient])


# PREDICT
disease_probs = {}

for d in LABELS:

    scaled = scalers[d].transform(input_df)

    prob = models[d].predict_proba(scaled)[0][1]

    disease_probs[d] = prob


# OUTPUT
best = max(disease_probs,key=disease_probs.get)

print("\nRESULT")
print("Predicted Disease:",LABELS[best])
print(f"Confidence: {disease_probs[best]*100:.2f}%")

print("\nALL PROBABILITIES:\n")

for d,p in sorted(disease_probs.items(),key=lambda x:x[1],reverse=True):
    print(f"{LABELS[d]:<20} {p*100:.2f}%")