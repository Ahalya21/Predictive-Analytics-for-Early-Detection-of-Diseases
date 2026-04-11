'''import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────
FEATURES = [
    'age','gender','bmi',
    'smoking','alcohol_consumption','physical_activity','diet_quality',
    'chest_pain','shortness_of_breath','coughing',
    'avg_glucose_level','HbA1c_level',
    'cholesterol_total','blood_pressure',
    'serum_creatinine',
]

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
alz    = pd.read_csv("alzheimers_disease_data.csv")
brca   = pd.read_csv("brca.csv")
diab   = pd.read_csv("diabetic_data.csv")
stroke = pd.read_csv("healthcare-dataset-stroke-data.csv")
heart  = pd.read_csv("HeartDiseaseTrain-Test.csv")
kidney = pd.read_csv("kidney-stone-dataset.csv")
lung   = pd.read_csv("lung_disease_data.csv")

# ─────────────────────────────────────────────
# SIMPLE FEATURE BUILDER (you can replace with your full mapping)
# ─────────────────────────────────────────────
def simple(df, disease):
    n = len(df)

    if disease == 'diabetic':
        return pd.DataFrame({
            'age': np.random.randint(40, 70, n),
            'gender': np.random.randint(0, 2, n),
            'bmi': np.random.normal(30, 4, n),
            'smoking': np.random.randint(0, 2, n),
            'alcohol_consumption': np.random.randint(1, 5, n),
            'physical_activity': np.random.randint(0, 3, n),
            'diet_quality': np.random.randint(2, 5, n),
            'chest_pain': 0,
            'shortness_of_breath': 0,
            'coughing': 0,
            'avg_glucose_level': np.random.normal(180, 30, n),  #  HIGH
            'HbA1c_level': np.random.normal(7.5, 1, n),         #  HIGH
            'cholesterol_total': np.random.normal(220, 20, n),
            'blood_pressure': np.random.normal(90, 10, n),
            'serum_creatinine': np.random.normal(1.2, 0.2, n),
            'disease': disease
        })

    elif disease == 'heart':
        return pd.DataFrame({
            'age': np.random.randint(50, 80, n),
            'gender': np.random.randint(0, 2, n),
            'bmi': np.random.normal(28, 3, n),
            'smoking': np.random.randint(1, 3, n),
            'alcohol_consumption': np.random.randint(3, 7, n),
            'physical_activity': np.random.randint(0, 3, n),
            'diet_quality': np.random.randint(2, 5, n),
            'chest_pain': 1,  #  KEY
            'shortness_of_breath': 1,
            'coughing': 0,
            'avg_glucose_level': np.random.normal(110, 20, n),
            'HbA1c_level': np.random.normal(5.8, 0.5, n),
            'cholesterol_total': np.random.normal(260, 30, n),  #  HIGH
            'blood_pressure': np.random.normal(100, 10, n),
            'serum_creatinine': np.random.normal(1.1, 0.2, n),
            'disease': disease
        })

    elif disease == 'lung':
        return pd.DataFrame({
            'age': np.random.randint(50, 75, n),
            'gender': np.random.randint(0, 2, n),
            'bmi': np.random.normal(26, 3, n),
            'smoking': 3,  #  VERY HIGH
            'alcohol_consumption': np.random.randint(3, 6, n),
            'physical_activity': np.random.randint(0, 3, n),
            'diet_quality': np.random.randint(2, 5, n),
            'chest_pain': 1,
            'shortness_of_breath': 1,
            'coughing': 1,  #  KEY
            'avg_glucose_level': np.random.normal(100, 15, n),
            'HbA1c_level': np.random.normal(5.5, 0.5, n),
            'cholesterol_total': np.random.normal(200, 20, n),
            'blood_pressure': np.random.normal(85, 10, n),
            'serum_creatinine': np.random.normal(1.0, 0.2, n),
            'disease': disease
        })

    elif disease == 'kidney':
        return pd.DataFrame({
            'age': np.random.randint(40, 70, n),
            'gender': np.random.randint(0, 2, n),
            'bmi': np.random.normal(27, 3, n),
            'smoking': np.random.randint(0, 2, n),
            'alcohol_consumption': np.random.randint(1, 4, n),
            'physical_activity': np.random.randint(1, 4, n),
            'diet_quality': np.random.randint(4, 7, n),
            'chest_pain': 0,
            'shortness_of_breath': 0,
            'coughing': 0,
            'avg_glucose_level': np.random.normal(120, 20, n),
            'HbA1c_level': np.random.normal(6, 0.5, n),
            'cholesterol_total': np.random.normal(200, 20, n),
            'blood_pressure': np.random.normal(95, 10, n),
            'serum_creatinine': np.random.normal(2.5, 0.5, n),  #  KEY
            'disease': disease
        })

    else:
        return pd.DataFrame({
            'age': np.random.randint(40, 70, n),
            'gender': np.random.randint(0, 2, n),
            'bmi': np.random.normal(25, 3, n),
            'smoking': np.random.randint(0, 2, n),
            'alcohol_consumption': np.random.randint(1, 5, n),
            'physical_activity': np.random.randint(2, 5, n),
            'diet_quality': np.random.randint(5, 8, n),
            'chest_pain': 0,
            'shortness_of_breath': 0,
            'coughing': 0,
            'avg_glucose_level': np.random.normal(100, 15, n),
            'HbA1c_level': np.random.normal(5.5, 0.5, n),
            'cholesterol_total': np.random.normal(190, 20, n),
            'blood_pressure': np.random.normal(80, 10, n),
            'serum_creatinine': np.random.normal(1.0, 0.2, n),
            'disease': disease
        })

alz_df    = simple(alz,'alzheimers')
brca_df   = simple(brca,'brca')
diab_df   = simple(diab,'diabetic')
stroke_df = simple(stroke,'stroke')
heart_df  = simple(heart,'heart')
kidney_df = simple(kidney,'kidney')
lung_df   = simple(lung,'lung')

combined = pd.concat([alz_df, brca_df, diab_df, stroke_df, heart_df, kidney_df, lung_df])
combined[FEATURES] = combined[FEATURES].fillna(
    combined[FEATURES].median()
)
# ─────────────────────────────────────────────
# SAVE MEDIANS + LABELS
# ─────────────────────────────────────────────
feature_medians = combined[FEATURES].median().to_dict()
joblib.dump(feature_medians, 'feature_medians.pkl')

le = LabelEncoder()
le.fit(combined['disease'])
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(FEATURES, 'feature_cols.pkl')

# ─────────────────────────────────────────────
# MODEL MAP (DIFFERENT MODEL PER DISEASE)
# ─────────────────────────────────────────────
MODEL_MAP = {
    'heart': RandomForestClassifier(n_estimators=300, random_state=42),

    'diabetic': lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03
    ),

    'stroke': LogisticRegression(max_iter=1000),

    'kidney': RandomForestClassifier(n_estimators=200),

    'lung': lgb.LGBMClassifier(n_estimators=300),

    'alzheimers': LogisticRegression(max_iter=1000),

    'brca': lgb.LGBMClassifier(n_estimators=500)
}

# ─────────────────────────────────────────────
# TRAIN SEPARATE MODELS
# ─────────────────────────────────────────────
print("\nTRAINING MODELS...\n")

for disease in combined['disease'].unique():
    print(f"Training {disease}")

    y = (combined['disease'] == disease).astype(int)
    X = combined[FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Select model
    model = MODEL_MAP.get(disease)

    if model is None:
        model = lgb.LGBMClassifier(n_estimators=200)

    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"{disease} accuracy: {acc:.3f}")

    # SAVE
    joblib.dump(model, f"disease_model_{disease}.pkl")
    joblib.dump(scaler, f"scaler_{disease}.pkl")

print("\nALL MODELS TRAINED & SAVED!")'''



















import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────
FEATURES = [
    'age','gender','bmi',
    'smoking','alcohol_consumption','physical_activity','diet_quality',
    'chest_pain','shortness_of_breath','coughing',
    'avg_glucose_level','HbA1c_level',
    'cholesterol_total','blood_pressure',
    'serum_creatinine',
]

# ─────────────────────────────────────────────
# LOAD CSV FILES
# ─────────────────────────────────────────────
alz    = pd.read_csv("alzheimers_disease_data.csv")
brca   = pd.read_csv("brca.csv")
diab   = pd.read_csv("diabetic_data.csv")
stroke = pd.read_csv("healthcare-dataset-stroke-data.csv")
heart  = pd.read_csv("HeartDiseaseTrain-Test.csv")
kidney = pd.read_csv("kidney-stone-dataset.csv")
lung   = pd.read_csv("lung_disease_data.csv")


# ─────────────────────────────────────────────
# SYNTHETIC FEATURE BUILDER
# ─────────────────────────────────────────────
def build_features(n, disease):

    if disease == 'diabetic':
        return pd.DataFrame({
            'age': np.random.randint(40,70,n),
            'gender': np.random.randint(0,2,n),
            'bmi': np.random.normal(31,4,n),
            'smoking': np.random.randint(0,2,n),
            'alcohol_consumption': np.random.randint(1,4,n),
            'physical_activity': np.random.randint(0,2,n),
            'diet_quality': np.random.randint(1,4,n),
            'chest_pain': 0,
            'shortness_of_breath': 0,
            'coughing': 0,
            'avg_glucose_level': np.random.normal(190,20,n),
            'HbA1c_level': np.random.normal(8,1,n),
            'cholesterol_total': np.random.normal(220,20,n),
            'blood_pressure': np.random.normal(95,10,n),
            'serum_creatinine': np.random.normal(1.3,0.2,n),
            'disease': disease
        })

    elif disease == 'heart':
         return pd.DataFrame({
            'age': np.random.randint(50,85,n),
            'gender': np.random.randint(0,2,n),
            'bmi': np.random.normal(30,3,n),

            'smoking': np.random.randint(1,3,n),
            'alcohol_consumption': np.random.randint(2,5,n),

            'physical_activity': np.random.randint(0,2,n),
            'diet_quality': np.random.randint(1,4,n),

            'chest_pain': np.random.binomial(1,0.85,n),
            'shortness_of_breath': np.random.binomial(1,0.75,n),
             'coughing': 0,

            'avg_glucose_level': np.random.normal(120,15,n),
            'HbA1c_level': np.random.normal(6,0.5,n),

            'cholesterol_total': np.random.normal(270,15,n),
            'blood_pressure': np.random.normal(145,10,n),

            'serum_creatinine': np.random.normal(1.1,0.2,n),

            'disease': disease
    })

    elif disease == 'stroke':
        return pd.DataFrame({
            'age': np.random.randint(55,85,n),
            'gender': np.random.randint(0,2,n),
            'bmi': np.random.normal(30,4,n),
            'smoking': np.random.randint(1,3,n),
            'alcohol_consumption': np.random.randint(2,5,n),
            'physical_activity': np.random.randint(0,2,n),
            'diet_quality': np.random.randint(1,4,n),
            'chest_pain': np.random.binomial(1,0.20,n),
            'shortness_of_breath': 1,
            'coughing': 0,
            'avg_glucose_level': np.random.normal(160,25,n),
            'HbA1c_level': np.random.normal(7,0.7,n),
            'cholesterol_total': np.random.normal(240,20,n),
            'blood_pressure': np.random.normal(140,10,n),
            'serum_creatinine': np.random.normal(1.2,0.2,n),
            'disease': disease
        })
    elif disease == 'alzheimers':
        return pd.DataFrame({
            'age': np.random.randint(70,95,n),
            'gender': np.random.randint(0,2,n),
            'bmi': np.random.normal(23,2,n),

            'smoking': np.random.randint(0,1,n),
            'alcohol_consumption': np.random.randint(0,2,n),

            'physical_activity': np.random.randint(0,1,n),
            'diet_quality': np.random.randint(3,5,n),

            'chest_pain': 0,
            'shortness_of_breath': 0,
            'coughing': 0,

            'avg_glucose_level': np.random.normal(95,10,n),
            'HbA1c_level': np.random.normal(5.4,0.3,n),

            'cholesterol_total': np.random.normal(185,10,n),
            'blood_pressure': np.random.normal(80,5,n),

            'serum_creatinine': np.random.normal(1.0,0.1,n),

            'disease': disease
    })

    elif disease == 'brca':
        return pd.DataFrame({
            'age': np.random.randint(35,65,n),
            'gender': 1,
            'bmi': np.random.normal(28,4,n),
            'smoking': np.random.randint(0,2,n),
            'alcohol_consumption': np.random.randint(1,4,n),
            'physical_activity': np.random.randint(1,4,n),
            'diet_quality': np.random.randint(3,5,n),
            'chest_pain': 0,
            'shortness_of_breath': 0,
            'coughing': 0,
            'avg_glucose_level': np.random.normal(105,10,n),
            'HbA1c_level': np.random.normal(5.7,0.3,n),
            'cholesterol_total': np.random.normal(195,15,n),
            'blood_pressure': np.random.normal(90,8,n),
            'serum_creatinine': np.random.normal(1.0,0.1,n),
            'disease': disease
        })

    elif disease == 'kidney':
        return pd.DataFrame({
            'age': np.random.randint(40,70,n),
            'gender': np.random.randint(0,2,n),
            'bmi': np.random.normal(27,3,n),
            'smoking': np.random.randint(0,2,n),
            'alcohol_consumption': np.random.randint(1,4,n),
            'physical_activity': np.random.randint(1,3,n),
            'diet_quality': np.random.randint(3,6,n),
            'chest_pain': 0,
            'shortness_of_breath': 0,
            'coughing': 0,
            'avg_glucose_level': np.random.normal(110,15,n),
            'HbA1c_level': np.random.normal(5.8,0.5,n),
            'cholesterol_total': np.random.normal(200,20,n),
            'blood_pressure': np.random.normal(95,8,n),
            'serum_creatinine': np.random.normal(2.8,0.4,n),
            'disease': disease
        })

    elif disease == 'lung':
        return pd.DataFrame({
            'age': np.random.randint(50,80,n),
            'gender': np.random.randint(0,2,n),
            'bmi': np.random.normal(25,3,n),
            'smoking': 3,
            'alcohol_consumption': np.random.randint(2,5,n),
            'physical_activity': np.random.randint(0,2,n),
            'diet_quality': np.random.randint(2,4,n),
            'chest_pain': 1,
            'shortness_of_breath': 1,
            'coughing': 1,
            'avg_glucose_level': np.random.normal(100,10,n),
            'HbA1c_level': np.random.normal(5.5,0.4,n),
            'cholesterol_total': np.random.normal(200,15,n),
            'blood_pressure': np.random.normal(90,8,n),
            'serum_creatinine': np.random.normal(1.0,0.1,n),
            'disease': disease
        })


# ─────────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────────
combined = pd.concat([
    build_features(len(alz),'alzheimers'),
    build_features(len(brca),'brca'),
    build_features(len(diab),'diabetic'),
    build_features(len(stroke),'stroke'),
    build_features(len(heart),'heart'),
    build_features(len(kidney),'kidney'),
    build_features(len(lung),'lung')
])

# SAVE MEDIANS
feature_medians = combined[FEATURES].median().to_dict()
joblib.dump(feature_medians,'feature_medians.pkl')
joblib.dump(FEATURES,'feature_cols.pkl')


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
MODEL_MAP = {
    'heart': RandomForestClassifier(n_estimators=300),
    'diabetic': lgb.LGBMClassifier(n_estimators=300),
    'stroke': LogisticRegression(max_iter=1000),
    'kidney': RandomForestClassifier(n_estimators=200),
    'lung': lgb.LGBMClassifier(n_estimators=300),
    'alzheimers': LogisticRegression(max_iter=1000),
    'brca': lgb.LGBMClassifier(n_estimators=300)
}

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
for disease in combined['disease'].unique():

    print(f"Training {disease}")

    X = combined[FEATURES]
    y = (combined['disease']==disease).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(
        X_scaled,y,test_size=0.2,stratify=y,random_state=42
    )

    model = MODEL_MAP[disease]
    model.fit(X_train,y_train)

    print("Accuracy:",model.score(X_test,y_test))

    joblib.dump(model,f"disease_model_{disease}.pkl")
    joblib.dump(scaler,f"scaler_{disease}.pkl")

print("\nALL MODELS TRAINED!")