import pandas as pd
import joblib

# ─────────────────────────────────────────────
# LOAD META & MODELS
# ─────────────────────────────────────────────
feature_cols    = joblib.load('feature_cols.pkl')
feature_medians = joblib.load('feature_medians.pkl')

LABELS = {
    'alzheimers': "Alzheimer's Disease",
    'brca':       "Breast Cancer",
    'diabetic':   "Diabetes",
    'heart':      "Heart Disease",
    'kidney':     "Kidney Disease",
    'lung':       "Lung Cancer",
    'stroke':     "Stroke",
}

models  = {}
scalers = {}

for d in LABELS:
    models[d]  = joblib.load(f"disease_model_{d}.pkl")
    scalers[d] = joblib.load(f"scaler_{d}.pkl")

# ─────────────────────────────────────────────
# COLLECT PATIENT INPUT
# ─────────────────────────────────────────────
print("\nEnter Patient Details:")
print("(Press Enter to use median/default value)\n")

FIELD_HINTS = {
    'age':                  'Age (years)',
    'gender':               'Gender (0=Male, 1=Female)',
    'bmi':                  'BMI',
    'smoking':              'Smoking (0=No, 1=Yes)',
    'alcohol_consumption':  'Alcohol Consumption (0-4)',
    'physical_activity':    'Physical Activity (1-4)',
    'diet_quality':         'Diet Quality (2-6)',
    'chest_pain':           'Chest Pain (0=No, 1=Yes)',
    'shortness_of_breath':  'Shortness of Breath (0=No, 1=Yes)',
    'coughing':             'Coughing (0=No, 1=Yes)',
    'avg_glucose_level':    'Average Glucose Level (mg/dL)',
    'HbA1c_level':          'HbA1c Level (%)',
    'cholesterol_total':    'Total Cholesterol (mg/dL)',
    'blood_pressure':       'Blood Pressure (mmHg, diastolic)',
    'serum_creatinine':     'Serum Creatinine (mg/dL)',
}

patient = {}
for f in feature_cols:
    hint    = FIELD_HINTS.get(f, f)
    default = round(feature_medians[f], 2)
    val     = input(f"  {hint} [default: {default}]: ").strip()
    patient[f] = float(val) if val != "" else feature_medians[f]

input_df = pd.DataFrame([patient])

# ─────────────────────────────────────────────
# PREDICT — each model is binary one-vs-rest
# predict_proba(...)[0][1] = P(has this disease)
# ─────────────────────────────────────────────
disease_probs = {}

for d in LABELS:
    # Each scaler was fit only on training data for that disease split
    scaled = scalers[d].transform(input_df[feature_cols])

    # [0][1] = probability of positive class (has the disease)
    prob = models[d].predict_proba(scaled)[0][1]

    # ── Post-prediction clinical adjustments ──
    # Breast cancer: biologically rare in males
    if d == 'brca' and patient['gender'] == 0:
        prob *= 0.15

    # Alzheimer's: onset almost never before 60
    if d == 'alzheimers' and patient['age'] < 60:
        prob *= 0.25

    # Heart disease: chest pain is a strong indicator
    if d == 'heart' and patient['chest_pain'] == 0:
        prob *= 0.65

    # Lung: very unlikely without coughing or smoking
    if d == 'lung' and patient['coughing'] == 0 and patient['smoking'] == 0:
        prob *= 0.30

    # Kidney: creatinine is the dominant marker
    # Kidney: strong boost for high creatinine
    if d == 'kidney':
        if patient['serum_creatinine'] > 3.0:
            prob *= 1.8   # strong kidney signal
        elif patient['serum_creatinine'] > 2.0:
            prob *= 1.4
        elif patient['serum_creatinine'] < 1.5:
            prob *= 0.4

    # Diabetes: low glucose makes it very unlikely
    if d == 'diabetic' and patient['avg_glucose_level'] < 120:
        prob *= 0.35

    # Stroke: almost always linked to high BP
    # if d == 'stroke' and patient['blood_pressure'] < 130:
        # prob *= 0.40
    if d == 'stroke':
        if patient['blood_pressure'] > 160:
            prob *= 0.15  # boost strong stroke signal
        elif patient['blood_pressure'] < 130:
            prob *= 0.4

        if patient['age'] > 70:
            prob *= 0.05
    prob = max(0, min(prob, 1.0))
    

    disease_probs[d] = round(prob, 4)
# force stroke priority in critical condition
if patient['blood_pressure'] > 170 and patient['age'] > 70:
    disease_probs['stroke'] += 0.20
    disease_probs['stroke'] = min(disease_probs['stroke'], 1.0)
# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
sorted_probs = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)
best, best_prob = sorted_probs[0]

print("\n" + "=" * 50)
print("  DISEASE PROBABILITY REPORT")
print("=" * 50)
print(f"\n  {'Disease':<25} {'Probability':>12}")
print("  " + "-" * 38)

for d, p in sorted_probs:
    bar   = "█" * int(p * 30)
    print(f"  {LABELS[d]:<25} {p*100:>6.2f}%  {bar}")

print("\n" + "=" * 50)

# Threshold: only report if at least one disease clears 30%
THRESHOLD = 0.25

if best_prob < THRESHOLD:
    print("\n  No strong disease match found.")
    print("  All disease probabilities are below 30%.")
    print("  Patient profile appears within normal range.")
else:
    print(f"\n  Predicted Disease : {LABELS[best]}")
    print(f"  Confidence        : {best_prob*100:.2f}%")

    # Show any other diseases also above threshold
    others = [(d, p) for d, p in sorted_probs[1:] if p >= THRESHOLD]
    if others:
        print(f"\n  Also flagged (≥30%):")
        for d, p in others:
            print(f"    • {LABELS[d]:<22} {p*100:.2f}%")

print("=" * 50)