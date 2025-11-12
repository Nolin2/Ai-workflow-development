# app.py (Step 4: Single Customer Review)

import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import plotly.express as px
import os

# --- Configuration & Setup (Previous code remains the same) ---
st.set_page_config(
    page_title="Seller Churn Reviewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Load Model and Artifacts (Previous code remains the same) ---
@st.cache_resource
def load_model_artifacts():
    # ... (Model loading logic using 'model.pkl' and 'columns.pkl')
    # NOTE: Keep the original implementation here
    try:
        model = joblib.load('model.pkl') 
        feature_columns = joblib.load('columns.pkl')
        if not isinstance(feature_columns, list):
            if hasattr(feature_columns, 'tolist'):
                feature_columns = feature_columns.tolist()
            else:
                st.error(f"Feature columns loaded but not in list format. Found type: {type(feature_columns)}")
                return None, []
        st.success("Machine learning model and features loaded successfully.")
        return model, feature_columns
    except FileNotFoundError:
        st.error("Error: Required model files (model.pkl or columns.pkl) not found. Please place them in the app directory.")
        return None, []
    except Exception as e:
        st.error(f"An unexpected error occurred during artifact loading: {e}")
        return None, []

model, feature_columns = load_model_artifacts()

# --- 2. Utility Functions (Previous code remains the same) ---

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Applies necessary transformations to the input data to match the model's required format.
    
    This implementation handles the 5 features found in 'columns.pkl'.
    """
    if not feature_columns: return None
    df_processed = df.copy()
    if 'Gender' in df_processed.columns:
        df_processed['Gender'] = df_processed['Gender'].astype(str).str.lower().map({'male': 1, 'female': 0}).fillna(0)
    for col in ['Age', 'Monthly_Spending', 'Subscription_Length', 'Support_Interactions']:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
    
    # Check for missing required features
    if not all(col in df_processed.columns for col in feature_columns):
        missing_cols = [col for col in feature_columns if col not in df_processed.columns]
        # In single prediction, we trust the input form provides all features, so we skip the error here.
        # This check is more critical for bulk upload.
        pass 

    final_df = df_processed[feature_columns]
    return final_df

def assign_priority(prob):
    # ... (Priority assignment logic)
    if prob > 0.75: return '🚨 High'
    elif prob > 0.5: return '⚠️ Medium'
    else: return '🟢 Low'

def predict_churn(df_original: pd.DataFrame) -> pd.DataFrame | None:
    # ... (Prediction logic)
    if model is None or df_original.empty: return None
    X = preprocess_data(df_original.copy())
    if X is None or X.empty: return None
    try:
        probabilities = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)
    except Exception as e:
        st.error(f"Prediction failed. Check your data or model: {e}")
        return None
    results_df = df_original.copy()
    results_df['Churn_Probability'] = probabilities.round(3)
    results_df['Churn_Prediction'] = np.where(predictions == 1, 'Yes', 'No')
    results_df['Retention_Priority'] = results_df['Churn_Probability'].apply(assign_priority)
    results_df['Churn_Icon'] = results_df['Churn_Prediction'].apply(lambda x: '❌' if x == 'Yes' else '✅')
    return results_df.sort_values(by='Churn_Probability', ascending=False)


# --- 3. Streamlit App Layout ---

st.title("🎯 Seller's Customer Churn Prediction Reviewer")
tab1, tab2, tab3 = st.tabs(["Upload & Predict", "Actionable Insights Dashboard", "Single Customer Review"])

# --- Tab 1 & 2 (Previous code remains the same) ---
with tab1:
    # ... (Content for tab 1 - Bulk Upload)
    pass # Placeholder to keep code focused on tab3

with tab2:
    # ... (Content for tab 2 - Dashboard)
    pass # Placeholder to keep code focused on tab3


# -----------------------------------------------------------
# --- Tab 3: Single Customer Review (NEW IMPLEMENTATION)
# -----------------------------------------------------------
with tab3:
    st.header("3. Review Single Customer Profile")
    st.info("Enter a customer's details below to get an instant churn prediction.")
    
    if model is None:
        st.error("Prediction service unavailable. Please check the model files.")
    else:
        with st.form("single_prediction_form"):
            
            # --- Input Layout ---
            col_a, col_b = st.columns(2)
            
            # Column A Inputs
            with col_a:
                age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30, step=1)
                monthly_spending = st.number_input("Monthly Spending ($)", min_value=0.0, value=50.0, step=0.5, format="%.2f")
                support_interactions = st.number_input("Support Interactions (Last 3 Months)", min_value=0, max_value=20, value=1, step=1)
            
            # Column B Inputs
            with col_b:
                gender = st.selectbox("Gender", options=['Female', 'Male'])
                subscription_length = st.number_input("Subscription Length (Months)", min_value=1, value=12, step=1)
            
            st.markdown("---")
            submit_button = st.form_submit_button("Get Instant Churn Prediction", type="primary")

            if submit_button:
                # 1. Prepare data for prediction
                input_data = {
                    'Age': age,
                    'Gender': gender,
                    'Monthly_Spending': monthly_spending,
                    'Subscription_Length': subscription_length,
                    'Support_Interactions': support_interactions
                }
                
                # Create a single-row DataFrame
                single_customer_df = pd.DataFrame([input_data])
                
                # 2. Predict (Use the existing utility function structure)
                single_result_df = predict_churn(single_customer_df)
                
                if single_result_df is not None:
                    prob = single_result_df['Churn_Probability'].iloc[0]
                    pred = single_result_df['Churn_Prediction'].iloc[0]
                    priority = single_result_df['Retention_Priority'].iloc[0]

                    st.subheader("Prediction Result")
                    
                    pred_col, prob_col, prio_col = st.columns(3)
                    
                    # Display prediction and color coding
                    if pred == 'Yes':
                        pred_col.error(f"Predicted Churn: **{pred} ❌**")
                    else:
                        pred_col.success(f"Predicted Churn: **{pred} ✅**")
                        
                    prob_col.metric("Churn Probability Score", f"{prob*100:.2f}%")
                    prio_col.info(f"Retention Priority: **{priority}**")

                    st.markdown("---")
                    st.subheader("💡 Key Retention Focus Areas")
                    st.write(f"""
                    Based on these inputs, a high **Monthly Spending** and low **Support Interactions** are often key churn drivers. 
                    The seller should focus on:
                    * **Value Reinforcement:** Highlighting the benefits this customer is getting for their spending.
                    * **Proactive Check-in:** Contacting them to ensure satisfaction, especially since their **Subscription Length** is ${subscription_length}$ months, which might align with a renewal period.
                    """)
