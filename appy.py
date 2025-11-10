import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. Configuration and Asset Loading ---

# Define the paths for the saved artifacts
MODEL_PATH = 'models/churn_predictor.pkl'
SCALER_PATH = 'models/scaler.pkl' 

# Define the exact order of features used in training (CRITICAL!)
FEATURE_COLS = ['Age', 'Gender', 'Monthly_Spending', 'Subscription_Length', 'Support_Interactions']

@st.cache_resource
def load_assets():
    """Loads the trained model and fitted scaler."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or Scaler not found. Ensure 'full_churn_workflow.py' (or your training scripts) has been run successfully.")
        return None, None

model, scaler = load_assets()

# --- 2. Streamlit Application Layout ---

st.set_page_config(page_title="Customer Churn Risk Predictor", layout="centered")

st.title("💡 Customer Churn Risk Predictor")
st.markdown("### Deployment Interface for Retention Team")

if model is not None and scaler is not None:
    # --- 3. Input Form (Sidebar) ---
    st.sidebar.header("Customer Profile Input")
    
    # Input fields for the 5 features
    age = st.sidebar.slider("1. Age (Years)", 18, 75, 45)
    
    # Gender is a binary feature (0 or 1)
    gender_map = st.sidebar.radio("2. Gender", options=[("Male", 0), ("Female", 1)], format_func=lambda x: x[0])[1]
    
    monthly_spending = st.sidebar.number_input("3. Monthly Spending ($)", 10.0, 500.0, 250.0, step=5.0)
    subscription_length = st.sidebar.slider("4. Subscription Length (Months)", 1, 12, 6)
    support_interactions = st.sidebar.slider("5. Support Interactions (Last 6 Months)", 0, 5, 2)
    
    # --- 4. Prediction Logic ---

    # Collect inputs into a DataFrame row
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_map],
        'Monthly_Spending': [monthly_spending],
        'Subscription_Length': [subscription_length],
        'Support_Interactions': [support_interactions]
    })
    
    # --- Preprocessing: Scale the input data using the loaded scaler ---
    
    # The scaler was fitted on all columns: Age, Gender, Monthly_Spending, Subscription_Length, Support_Interactions.
    # We must transform the input data exactly as the training data was transformed.
    
    input_scaled_values = scaler.transform(input_data[FEATURE_COLS])
    
    # Convert scaled values back to DataFrame in the correct order for prediction
    final_features_df = pd.DataFrame(input_scaled_values, columns=FEATURE_COLS)

    # Make prediction (predict_proba gives the probability of each class)
    prediction_proba = model.predict_proba(final_features_df.values)[0]
    churn_proba = prediction_proba[1] # Probability of Churn (Class 1)

    # --- 5. Display Results ---

    st.subheader("🎯 Prediction Results")

    # Define risk levels based on probability
    if churn_proba >= 0.70:
        risk_level = "High Risk - IMMEDIATE INTERVENTION REQUIRED!"
        color = "red"
        icon = "🚨"
    elif churn_proba >= 0.40:
        risk_level = "Medium Risk - Proactive Offer Needed"
        color = "orange"
        icon = "⚠️"
    else:
        risk_level = "Low Risk - Stable Customer"
        color = "green"
        icon = ""
    
    st.metric(
        label="Churn Probability Score", 
        value=f"{churn_proba:.2f}",
        delta=f"Risk: {risk_level}",
        delta_color=color
    )

    st.markdown(f"**Retention Strategy:** {icon} <span style='color:{color}; font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
    
    st.write("---")

    # Display Actionable Insights (Simulating Feature Importance)
    st.subheader(" Actionable Insights (Why This Score?)")
    
    if support_interactions >= 3 and monthly_spending < 300:
        st.error(f"""
        The **High/Medium** risk score is likely driven by the **combination** of:
        - **Support Interactions:** High count ({support_interactions}) suggests recent frustration or issues.
        - **Monthly Spending:** Spending (${monthly_spending:.0f}) is below average, indicating lower value derived from the service.
        """)
    elif subscription_length <= 3:
        st.warning(f"""
        The risk is elevated because of **Low Subscription Length** ({subscription_length} months). New customers are typically volatile. Flag for onboarding check-in.
        """)
    else:
        st.success("""
        The **Low** risk score is supported by consistent tenure and average product interaction. This customer appears stable.
        """)

else:
    st.warning("Application not ready. Please run your training script first to create the model artifacts!")
