# app.py (Step 1: Model Loading and Preprocessing)

import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import plotly.express as px

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Seller Churn Reviewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Load Model and Artifacts ---

@st.cache_resource
def load_model_artifacts():
    """Loads the ML model and feature list, caching them for performance."""
    try:
        # NOTE: Update file names if yours are different
        model = joblib.load('churn_model.pkl') 
        feature_columns = joblib.load('feature_columns.pkl')
        st.success("Machine learning model loaded successfully.")
        return model, feature_columns
    except FileNotFoundError:
        st.error("Error: Required model files (churn_model.pkl or feature_columns.pkl) not found. Please place them in the app directory.")
        return None, []

model, feature_columns = load_model_artifacts()

# --- 2. Utility Functions ---

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Applies necessary transformations (encoding, column alignment) to the input data.
    
    NOTE: This logic MUST mirror the steps used during model training!
    This example uses common Telecom Churn features for demonstration.
    """
    if not feature_columns:
        return None
    
    df_processed = df.copy()

    # --- A. Handle Categorical Features (One-Hot Encoding) ---
    # Common categorical features in Churn prediction:
    categorical_cols = ['Contract', 'PaymentMethod', 'Gender', 'MultipleLines', 'InternetService']

    # Convert object/string columns to category type first
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('category')
        
    # Apply One-Hot Encoding
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

    # --- B. Handle Numerical Features (e.g., Coercing TotalCharges) ---
    # TotalCharges often comes in as an object/string due to blanks/spaces
    if 'TotalCharges' in df_processed.columns:
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
        # Simple imputation for any resulting NaNs (e.g., fill with mean or 0)
        df_processed['TotalCharges'] = df_processed['TotalCharges'].fillna(0)
    
    # --- C. Align Columns with the Trained Model ---
    # This is the most crucial step! It ensures the new data has the same columns 
    # in the same order as the training data, padding missing columns with 0.
    
    # Create a DataFrame with all expected feature columns, initially filled with 0
    final_df = pd.DataFrame(0, index=df_processed.index, columns=feature_columns)
    
    # Copy the values from the processed input data where columns match
    available_cols = list(set(df_processed.columns) & set(feature_columns))
    final_df[available_cols] = df_processed[available_cols]

    return final_df

def predict_churn(df_original: pd.DataFrame) -> pd.DataFrame | None:
    """Makes predictions and adds results to the original DataFrame."""
    if model is None or df_original.empty:
        return None

    # 1. Preprocess the uploaded data
    X = preprocess_data(df_original.copy())
    if X is None or X.empty:
        st.error("Preprocessing failed or resulted in empty data.")
        return None

    # 2. Make predictions
    try:
        probabilities = model.predict_proba(X)[:, 1] # Probability of Churn (class 1)
        predictions = model.predict(X) # Hard Churn prediction (0 or 1)
    except Exception as e:
        st.error(f"Prediction failed. Check that your model features ({len(feature_columns)} expected) match the processed input: {e}")
        return None

    # 3. Combine results with original data
    results_df = df_original.copy()
    results_df['Churn_Probability'] = probabilities.round(3)
    results_df['Churn_Prediction'] = np.where(predictions == 1, 'Yes', 'No')

    # 4. Determine Seller Action Priority
    def assign_priority(prob):
        if prob > 0.75:
            return '🚨 High'
        elif prob > 0.5:
            return '⚠️ Medium'
        else:
            return '🟢 Low'

    results_df['Retention_Priority'] = results_df['Churn_Probability'].apply(assign_priority)

    return results_df.sort_values(by='Churn_Probability', ascending=False)
    
# Placeholder for the Streamlit UI layout (will be developed in Step 2)
# st.title("🎯 Seller's Customer Churn Prediction Reviewer")
# # ... (rest of the UI code)
