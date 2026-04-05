
import numpy as np
import pandas as pd
import warnings
import joblib

# ── LOAD ────────────────────────────────────────────────────────────────
model        = joblib.load('disease_model.pkl')
scaler       = joblib.load('scaler.pkl')
le           = joblib.load('label_encoder.pkl')
feature_cols = joblib.load('feature_cols.pkl')

LABELS = {
    'alzheimers': "Alzheimer's disease",
    'brca':       "Breast cancer",
    'diabetic':   "Diabetes",
    'heart':      "Heart disease",
    'kidney':     "Kidney disease",
    'lung':       "Lung cancer",
    'stroke':     "Stroke",
}

TIPS = {
    'alzheimers': "Age, poor diet, low activity and high alcohol are key risk factors.",
    'brca':       "Tumor size and shape are the strongest indicators. Seek screening.",
    'diabetic':   "HbA1c and glucose levels are the most critical markers.",
    'heart':      "Cholesterol, chest pain type and max heart rate are key signals.",
    'kidney':     "Creatinine and blood urea are the strongest kidney disease markers.",
    'lung':       "Smoking history combined with coughing and breathlessness is a strong signal.",
    'stroke':     "Age, BMI, hypertension and glucose are the leading stroke risk factors.",
}

# ── QUESTIONS ───────────────────────────────────────────────────────────
# Only general fields — no disease-specific lab or pathology values.
QUESTIONS = [
    ('__section__', 'DEMOGRAPHICS', '', ''),
    ('age',    'Age',    'years, e.g. 45',  'number'),
    ('gender', 'Gender', 'male / female',   'gender'),
    ('bmi',    'BMI',    'e.g. 24.5',       'number'),

    ('__section__', 'LIFESTYLE', '', ''),
    ('smoking',             'Smoking',            '0=never  1=former  2=current  3=heavy', 'number'),
    ('alcohol_consumption', 'Alcohol consumption','units per week, e.g. 3',                'number'),
    ('physical_activity',   'Physical activity',  'hours per week, e.g. 4',               'number'),
    ('diet_quality',        'Diet quality',       '0=very poor  →  10=excellent',         'number'),

    ('__section__', 'SYMPTOMS', '', ''),
    ('chest_pain',          'Chest pain',          'yes / no', 'binary'),
    ('shortness_of_breath', 'Shortness of breath', 'yes / no', 'binary'),
    ('coughing',            'Persistent coughing', 'yes / no', 'binary'),

    ('__section__', 'KEY LAB VALUES  (press Enter to skip any unknown)', '', ''),
    ('avg_glucose_level', 'Blood glucose level', 'mg/dL, e.g. 110  — most important for diabetes & stroke', 'number'),
    ('HbA1c_level',       'HbA1c level',         '%, e.g. 6.5  — key for diabetes',                        'number'),
    ('cholesterol_total', 'Total cholesterol',   'mg/dL, e.g. 200  — key for heart & alzheimers',          'number'),
    ('serum_creatinine',  'Serum creatinine',    'mg/dL, e.g. 1.1  — key for kidney disease',              'number'),
    ('blood_pressure',    'Blood pressure',      'mmHg diastolic, e.g. 80  — key for kidney & stroke',     'number'),
]


# ── INPUT HELPER ────────────────────────────────────────────────────────
def ask(label, hint, input_type):
    raw = input(f"  {label:<36}  ({hint}): ").strip().lower()
    if not raw:
        return 0.0
    if input_type == 'gender':
        return 1.0 if raw in ('male', 'm', '1') else 0.0
    if input_type == 'binary':
        return 1.0 if raw in ('yes', 'y', '1') else 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


# ── HARD-RULE OVERRIDE ───────────────────────────────────────────────────
# Only rules based on general lab values that are actually collected.
# Returns (disease_key, reason_string) if a hard rule fires, else None.
def check_hard_rules(p):
    creat   = p.get('serum_creatinine',  0.0)
    hba1c   = p.get('HbA1c_level',       0.0)
    glucose = p.get('avg_glucose_level', 0.0)

    # Kidney disease — creatinine critically high
    if creat > 2.0:
        return ('kidney', f'Serum creatinine {creat} mg/dL is critically elevated (normal < 1.2)')

    # Diabetes — both HbA1c and glucose in diabetic range
    if hba1c > 6.5 and glucose > 140:
        return ('diabetic', f'HbA1c {hba1c}% + glucose {glucose} mg/dL both in diabetic range')

    return None   # no hard rule fired — use model prediction


# ── HEADER ──────────────────────────────────────────────────────────────
print("  Fill in what you know. Press Enter to skip any field.")
print()

# ── COLLECT ALL ANSWERS ─────────────────────────────────────────────────
patient = {f: 0.0 for f in feature_cols}

for feat, label, hint, itype in QUESTIONS:
    if feat == '__section__':
        print(f"\n  -- {label} {'-' * (48 - len(label))}")
        continue
    if feat not in patient:
        continue
    patient[feat] = ask(label, hint, itype)


# ── HARD-RULE CHECK ──────────────────────────────────────────────────────
override = check_hard_rules(patient)

warnings.filterwarnings('ignore')   # suppress sklearn feature-name warning

# Build input as DataFrame with correct column names
input_df  = pd.DataFrame([[patient[f] for f in feature_cols]], columns=feature_cols)
scaled    = scaler.transform(input_df)
scaled_df = pd.DataFrame(scaled, columns=feature_cols)

if override:
    # Hard rule fired — skip model prediction, use rule result
    disease, reason = override
    confidence      = 95.0
    override_fired  = True

    # Run model only for informational bar chart
    pred_proba = model.predict_proba(scaled_df)[0]

else:
    # Normal model prediction
    override_fired = False
    pred_int   = model.predict(scaled_df)[0]
    pred_proba = model.predict_proba(scaled_df)[0]
    disease    = le.inverse_transform([pred_int])[0]
    confidence = round(pred_proba[pred_int] * 100, 1)


# ── DISPLAY RESULT ───────────────────────────────────────────────────────
print()
print(f"  PREDICTED DISEASE  :  {LABELS.get(disease, disease)}")
print(f"  CONFIDENCE         :  {confidence:.1f}%")

if override_fired:
    print(f"\n  [CLINICAL OVERRIDE]  Rule fired:")
    print(f"  {reason}")
    print(f"  Model prediction overridden — lab values are definitive.")

print()
print(f"  Tip: {TIPS.get(disease, '')}")
print()

# All 7 diseases — always shown
print("  Risk probabilities across all 7 diseases:\n")
BAR = 30

# Build display probabilities for the bar chart
display_proba = list(zip(le.classes_, pred_proba))
if override_fired:
    disease_idx = list(le.classes_).index(disease)
    boosted     = pred_proba.copy()
    others_sum  = sum(v for i, v in enumerate(boosted) if i != disease_idx)
    for i in range(len(boosted)):
        if i == disease_idx:
            boosted[i] = 0.95
        else:
            boosted[i] = (boosted[i] / others_sum) * 0.05 if others_sum > 0 else 0.05 / (len(boosted) - 1)
    display_proba = list(zip(le.classes_, boosted))

sorted_p = sorted(display_proba, key=lambda x: x[1], reverse=True)
for cls, prob in sorted_p:
    name   = LABELS.get(cls, cls)
    filled = int(prob * BAR)
    bar    = '█' * filled + '░' * (BAR - filled)
    arrow  = '  <- predicted' if cls == disease else ''
    print(f"  {name:<25}  {prob*100:>5.1f}%  [{bar}]{arrow}")

print()

if not override_fired and confidence < 45:
    print("  Low confidence — profile matches multiple diseases.")
    print("  Try entering more lab values for a more precise result.")
    print()

print("  " + "-" * 52)
print("  Screening tool only. Consult a doctor for diagnosis.")
print()