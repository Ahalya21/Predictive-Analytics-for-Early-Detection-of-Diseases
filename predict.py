# predict_terminal.py
import pandas as pd
import joblib

model        = joblib.load('disease_model.pkl')
scaler       = joblib.load('scaler.pkl')
le           = joblib.load('label_encoder.pkl')
top_features = joblib.load('top_features.pkl')

print("\n  === Disease Prediction ===")
print("  Enter patient values (press Enter to skip)\n")

# Questions to ask in terminal
QUESTIONS = {
    'age':               'Age (years)',
    'bmi':               'BMI',
    'cholesterol':       'Cholesterol (mg/dL)',
    'systolic_bp':       'Systolic BP (mmHg)',
    'glucose':           'Glucose (mg/dL)',
    'smoking':           'Smoking? (1=Yes, 0=No)',
    'hypertension':      'Hypertension? (1=Yes, 0=No)',
    'diabetes':          'Diabetes? (1=Yes, 0=No)',
    'cardiovascular':    'Cardiovascular disease? (1=Yes, 0=No)',
    'physical_activity': 'Physical activity (hrs/week)',
    'max_heart_rate':    'Max heart rate (bpm)',
    'lung_capacity':     'Lung capacity (L)',
}

patient = {}
for key, question in QUESTIONS.items():
    val = input(f"  {question}: ").strip()
    patient[key] = float(val) if val else 0.0

# Run prediction
df = pd.DataFrame([patient])
for col in top_features:
    if col not in df.columns:
        df[col] = 0
df = df[top_features]
df_scaled = scaler.transform(df)

pred_int   = model.predict(df_scaled)[0]
pred_proba = model.predict_proba(df_scaled)[0]
disease    = le.inverse_transform([pred_int])[0]
confidence = round(pred_proba[pred_int] * 100, 1)

print(f"\n  Predicted : {disease.upper()}  ({confidence}% confidence)\n")