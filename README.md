# 🏥 Multi-Disease Prediction System

A machine learning system that predicts the likelihood of **7 major diseases** from a patient's clinical and lifestyle profile. Uses an ensemble of calibrated classifiers (Random Forest, LightGBM, Gradient Boosting) in a one-vs-rest architecture to output per-disease probability scores.

---

## 📋 Diseases Covered

| Disease | Key Indicators |
|---|---|
| ❤️ Heart Disease | Chest pain, cholesterol, blood pressure |
| 🩸 Diabetes | Glucose level, HbA1c, BMI |
| 🧠 Stroke | Blood pressure, age, glucose |
| 🫁 Lung Cancer | Coughing, smoking, shortness of breath |
| 🫀 Kidney Disease | Serum creatinine, blood pressure |
| 🧬 Alzheimer's Disease | Age, physical activity, diet quality |
| 🎗️ Breast Cancer | Gender, BMI, age, alcohol consumption |

---

## 🧠 How It Works

### Architecture
Each disease has its own **binary classifier** trained in a one-vs-rest setup — meaning it learns to distinguish "has disease X" vs. "does not have disease X." All classifiers are wrapped in **CalibratedClassifierCV** (sigmoid calibration) to produce reliable probability estimates.

### Models Used
- **Random Forest** — Heart Disease, Kidney Disease
- **LightGBM** — Diabetes, Lung Cancer
- **Gradient Boosting** — Stroke, Alzheimer's, Breast Cancer

### Post-Prediction Clinical Adjustments
Raw model probabilities are adjusted using domain knowledge rules, for example:
- Breast cancer probability reduced by 85% for male patients
- Alzheimer's probability reduced by 75% for patients under 60
- Kidney disease boosted for high serum creatinine (> 3.0 mg/dL)
- Stroke boosted for very high blood pressure (> 170 mmHg) combined with age > 70

---

## 📁 Project Structure

```
├── train_model.py          # Synthetic data generation + model training
├── predict.py              # CLI patient intake + disease prediction
├── feature_cols.pkl        # Saved list of feature column names
├── feature_medians.pkl     # Saved median values for default imputation
├── disease_model_*.pkl     # Trained model for each disease (7 files)
└── scaler_*.pkl            # StandardScaler for each disease (7 files)
```

---

## ⚙️ Setup & Installation

**Prerequisites:** Python 3.8+

```bash
pip install pandas numpy scikit-learn lightgbm joblib
```

---

## 🚀 Usage

### Step 1 — Train the Models

```bash
python train_model.py
```

This will:
- Generate a balanced synthetic dataset of 7 × N samples (where N = min 5000, capped by the smallest real dataset)
- Train and calibrate one model per disease
- Save all models, scalers, and metadata as `.pkl` files

### Step 2 — Run Predictions

```bash
python predict.py
```

You will be prompted to enter patient values one by one. Press **Enter** to accept the median default for any field.

#### Example Input Fields

| Field | Description | Range |
|---|---|---|
| `age` | Patient age in years | 30–80 |
| `gender` | 0 = Male, 1 = Female | 0 / 1 |
| `bmi` | Body Mass Index | Numeric |
| `smoking` | Smoker? | 0 / 1 |
| `alcohol_consumption` | Alcohol use level | 0–4 |
| `physical_activity` | Activity level | 1–4 |
| `diet_quality` | Diet quality score | 2–6 |
| `chest_pain` | Chest pain present? | 0 / 1 |
| `shortness_of_breath` | Shortness of breath? | 0 / 1 |
| `coughing` | Coughing present? | 0 / 1 |
| `avg_glucose_level` | Average blood glucose | mg/dL |
| `HbA1c_level` | Glycated hemoglobin | % |
| `cholesterol_total` | Total cholesterol | mg/dL |
| `blood_pressure` | Diastolic blood pressure | mmHg |
| `serum_creatinine` | Kidney filtration marker | mg/dL |

#### Example Output

```
==================================================
  DISEASE PROBABILITY REPORT
==================================================

  Disease                   Probability
  --------------------------------------
  Stroke                      72.14%  █████████████████████
  Heart Disease               45.30%  █████████████
  Diabetes                    21.05%  ██████

==================================================
  Predicted Disease : Stroke
  Confidence        : 72.14%

  Also flagged (≥30%):
    • Heart Disease            45.30%
==================================================
```

Diseases with probability below **25%** are not flagged as predictions.

---

## 📊 Features

- **15 clinical + lifestyle features** per patient
- **Balanced training** via `class_weight='balanced'` and stratified splits
- **Noise injection** during data generation for more robust training
- **Calibrated probabilities** — scores reflect true likelihood, not just rank
- **Domain-rule post-processing** for clinically implausible outputs

---

## ⚠️ Disclaimer

This system is built on **synthetically generated training data** designed to mimic real-world disease distributions. It is intended as a **demonstration and educational tool only** and should **not** be used for real medical diagnosis or clinical decision-making. Always consult a qualified healthcare professional.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.