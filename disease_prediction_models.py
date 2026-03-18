import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# LOAD DATASETS
lung_df = pd.read_csv("lung_disease_data.csv")
kidney_df = pd.read_csv("kidney-stone-dataset.csv")
heart_df = pd.read_csv("HeartDiseaseTrain-Test.csv")
stroke_df = pd.read_csv("healthcare-dataset-stroke-data.csv")
diabetes_df = pd.read_csv("diabetic_data.csv")
cancer_df = pd.read_csv("brca.csv")
alz_df = pd.read_csv("alzheimers_disease_data.csv")


# TRAINING FUNCTION
def train_model(df, target_column, model_name):

    print(f"\n Training model for {model_name}")

    # Drop unwanted columns
    df = df.drop(columns=[
        "Unnamed: 0", "id", "PatientID",
        "encounter_id", "patient_nbr"
    ], errors="ignore")

    # Handle missing values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode target column if categorical
    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].astype('category').cat.codes

    # Split features & target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # One-hot encoding
    X = pd.get_dummies(X)

    # Save feature columns
    feature_columns = X.columns

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}: {acc:.4f}")

    # Save model and columns
    joblib.dump(model, f"{model_name}_model.pkl")
    joblib.dump(feature_columns, f"{model_name}_columns.pkl")

    print(f" Saved {model_name}_model.pkl and columns")

    return model


# TRAIN ALL MODELS

# IMPORTANT: Update target columns if needed

heart_model = train_model(heart_df, "target", "heart")

diabetes_model = train_model(diabetes_df, "readmitted", "diabetes")

stroke_model = train_model(stroke_df, "stroke", "stroke")

# Update below targets based on your dataset columns
lung_model = train_model(lung_df, "target", "lung")          # change if needed
kidney_model = train_model(kidney_df, "target", "kidney")    # change if needed
cancer_model = train_model(cancer_df, "target", "cancer")    # change if needed
alz_model = train_model(alz_df, "target", "alzheimers")      # change if needed


# PREDICTION FUNCTI
def predict_disease(model_name, input_dict):

    # Load model and columns
    model = joblib.load(f"{model_name}_model.pkl")
    columns = joblib.load(f"{model_name}_columns.pkl")

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])

    # One-hot encode input
    input_df = pd.get_dummies(input_df)

    # Align columns with training data
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Predict
    prediction = model.predict(input_df)

    return prediction[0]


# EXAMPLE PREDICTION
# Example (update with real features)
# sample_input = {
#     "age": 55,
#     "sex": 1,
#     "cp": 2,
#     "trestbps": 130,
#     "chol": 250
# }

# result = predict_disease("heart", sample_input)
# print("Prediction:", result)