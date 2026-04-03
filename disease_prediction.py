"""
predict.py  —  Single-stage disease prediction
════════════════════════════════════════════════
FIX 3 applied here:
  Hard-rule overrides BEFORE the model runs.
  If the patient clearly has disease-specific lab evidence,
  the model prediction is overridden with certainty.

  Rules:
    tumor_radius > 10  AND  tumor_area > 200   → Breast cancer  (override)
    serum_creatinine > 2.0  OR  blood_urea > 60 → Kidney disease (override)
    HbA1c > 6.5  AND  avg_glucose > 140         → Diabetes       (override)
    lung_capacity < 3.0  AND  (smoking>1 OR coughing) → Lung cancer (override)
"""

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

    ('__section__', 'DISEASE-SPECIFIC  (press Enter to skip if not applicable)', '', ''),
    ('functional_assessment', 'Functional assessment score', '0-10  (cognitive ability — alzheimers)',           'number'),
    ('adl_score',             'ADL score',                   '0-10  (daily living ability — alzheimers)',        'number'),
    ('max_heart_rate',        'Max heart rate achieved',     'bpm, e.g. 150  (heart disease)',                   'number'),
    ('st_depression',         'ST depression',               'e.g. 1.5  (heart disease, skip if no ECG)',        'number'),
    ('lung_capacity',         'Lung capacity',               'liters, e.g. 4.5  (lung cancer)',                  'number'),
    ('blood_urea',            'Blood urea',                  'mg/dL, e.g. 40  (kidney disease)',                 'number'),
    ('albumin',               'Albumin in urine',            '0-5 scale  (kidney disease)',                      'number'),
    ('tumor_radius',          'Tumor radius mean',           'from pathology report, e.g. 14.5  (breast cancer)','number'),
    ('tumor_area',            'Tumor area mean',             'from pathology report, e.g. 650  (breast cancer)', 'number'),
    ('tumor_concavity',       'Tumor concavity mean',        'from pathology report, e.g. 0.3  (breast cancer)', 'number'),
    ('hypertension',          'Hypertension (high BP)',      'yes / no  (stroke & kidney)',                      'binary'),
    ('heart_disease_history', 'Heart disease history',       'yes / no  (diabetes & stroke)',                    'binary'),
    ('ever_married',          'Ever married',                'yes / no  (stroke risk factor)',                    'binary'),
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


# ── HARD-RULE OVERRIDE  (FIX 3) ─────────────────────────────────────────
# Returns (disease_key, reason_string) if a hard rule fires, else None.
# These run BEFORE the model — clear clinical evidence overrides ML.
def check_hard_rules(p):
    tumor_r = p.get('tumor_radius',    0.0)
    tumor_a = p.get('tumor_area',      0.0)
    tumor_c = p.get('tumor_concavity', 0.0)
    creat   = p.get('serum_creatinine',0.0)
    urea    = p.get('blood_urea',      0.0)
    hba1c   = p.get('HbA1c_level',    0.0)
    glucose = p.get('avg_glucose_level',0.0)
    lung_c  = p.get('lung_capacity',  0.0)
    smoking = p.get('smoking',         0.0)
    cough   = p.get('coughing',        0.0)

    # Breast cancer — tumor data present → near-certain
    if tumor_r > 10 and tumor_a > 200:
        return ('brca', f'Tumor radius {tumor_r} + area {tumor_a} strongly indicate breast cancer')

    # Breast cancer — even one strong tumor signal
    if tumor_r > 15 or tumor_a > 700 or tumor_c > 0.25:
        return ('brca', f'Tumor marker (radius={tumor_r}, area={tumor_a}, concavity={tumor_c}) detected')

    # Kidney disease — creatinine or urea critically high
    if creat > 2.0:
        return ('kidney', f'Serum creatinine {creat} mg/dL is critically elevated (normal < 1.2)')
    if urea > 60:
        return ('kidney', f'Blood urea {urea} mg/dL is critically elevated (normal < 40)')

    # Diabetes — both HbA1c and glucose in diabetic range
    if hba1c > 6.5 and glucose > 140:
        return ('diabetic', f'HbA1c {hba1c}% + glucose {glucose} mg/dL both in diabetic range')

    # Lung cancer — low capacity + smoking or coughing
    if 0 < lung_c < 3.0 and (smoking > 1 or cough > 0):
        return ('lung', f'Lung capacity {lung_c}L is critically low with smoking/cough history')

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

# Compute tumor_score for the model (same formula used during training)
patient['tumor_score'] = patient.get('tumor_radius', 0.0) * patient.get('tumor_concavity', 0.0)


# ── HARD-RULE CHECK  (FIX 3) ────────────────────────────────────────────
override = check_hard_rules(patient)

import pandas as pd
warnings.filterwarnings('ignore')   # suppress sklearn feature-name warning

# Build input as DataFrame with correct column names (fixes sklearn warning)
input_df = pd.DataFrame([[patient[f] for f in feature_cols]], columns=feature_cols)
scaled   = scaler.transform(input_df)
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
    # Fix Bug 1: set override disease to exactly 95%, share remaining 5% among others
    disease_idx  = list(le.classes_).index(disease)
    boosted      = pred_proba.copy()
    others_sum   = sum(v for i, v in enumerate(boosted) if i != disease_idx)
    # Scale other diseases to share the remaining 5%
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