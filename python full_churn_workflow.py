import pandas as pd
import numpy as np 
import joblib
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report, accuracy_score

# --- 1. Configuration and Paths ---
DATA_PATH = 'data/raw/customer_churn_dataset.csv'
MODEL_PATH = 'models/churn_predictor.pkl'
SCALER_PATH = 'models/scaler.pkl'
TEST_SIZE = 0.20
RANDOM_SEED = 42

print("--- Starting Full Churn Prediction Workflow ---")

# ==============================================================================
# I. DATA PREPARATION (Load, Scale, Split)
# ==============================================================================

print("\n--- I. Data Preparation ---")

# 1. Load Data
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully. Total samples: {len(df)}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Ensure the file is in data/raw/")
    exit()

# 2. Separate Features (X) and Target (y)
X = df.drop(['Customer_ID', 'Churn'], axis=1)
y = df['Churn']

# 3. Scaling: Fit scaler and transform features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_processed = pd.DataFrame(X_scaled, columns=X.columns)
print("Features scaled (StandardScaler applied).")

# Save the fitted scaler (CRUCIAL for consistent scaling in the Streamlit app)
joblib.dump(scaler, SCALER_PATH)
print(f"Fitted scaler successfully saved to {SCALER_PATH}")

# 4. Stratified Split (Ensures equal churn ratio in train/test)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"Data split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")

# ==============================================================================
# II. MODEL TRAINING (Train and Save)
# ==============================================================================

print("\n--- II. Model Training ---")

# 1. Model Definition (Random Forest Classifier as per notebook reference)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_SEED,
    # CRITICAL: Balances class weights to prioritize the minority class (Churn=1)
    class_weight='balanced',
    max_depth=10 
)

# 2. Train Model
print("Starting model training (with balanced class weights)...")
model.fit(X_train, y_train)
print("Model training complete.")

# 3. Save Model
joblib.dump(model, MODEL_PATH)
print(f"Trained model successfully saved to {MODEL_PATH}")

# ==============================================================================
# III. MODEL EVALUATION (Predict and Score)
# ==============================================================================

print("\n--- III. Model Evaluation ---")

# 1. Make Predictions on the Test Set
y_pred = model.predict(X_test)

# 2. Calculate Metrics
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
accuracy = accuracy_score(y_test, y_pred) 

# 3. Print Results (Content for your report's screenshot)
print("\n--- Final Model Performance Metrics ---")

print("\nConfusion Matrix (for Report):")
print(cm)

print("\nKey Business Metrics:")
print(f"Accuracy Score: {accuracy:.4f}")
print(f"Precision Score (Efficiency): {precision:.4f}")
print(f"Recall Score (Coverage - Primary Goal): {recall:.4f}")

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n--- Workflow Complete ---")

