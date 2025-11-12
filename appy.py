import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. Configuration and Asset Loading ---

# Define the paths for the saved artifacts (MUST match the paths in your training script)
MODEL_PATH = 'models/churn_predictor.pkl'
SCALER_PATH = 'models/scaler.pkl' 

# Define the exact order of features used in training (CRITICAL!)
FEATURE_COLS = ['Age', 'Gender', 'Monthly_Spending', 'Subscription_Length', 'Support_Interactions']

@st.cache_resource
def load_assets():
    """Loads the trained model and fitted scaler, using caching for performance."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        # This error message is visible on the main page if files are missing
        st.error("🚨 ERROR: Model or Scaler not found.")
        st.warning("Please run your training script (`full_churn_workflow.py`) first to generate the necessary files in the 'models/' folder.")
        return None, None

# Load the essential components
model, scaler = load_assets()

# --- 2. Streamlit Application Layout ---

st.set_page_config(page_title="Customer Churn Risk Predictor", layout="centered")

st.title("💡 Customer Churn Risk Predictor")
st.markdown("### Interactive Deployment Interface for Retention Team")

# Only render the UI if the assets loaded successfully
if model is not None and scaler is not None:
    
    # ====================================================================
    # USER INTERFACE: SIDEBAR INPUTS
    # ====================================================================
    st.sidebar.header("📊 Customer Profile Input")
    st.sidebar.markdown("Adjust the variables below to simulate a customer's risk profile.")
    
    # 1. Age (Slider)
    age = st.sidebar.slider("1. Age (Years)", 18, 75, 45, help="Customer's current age.")
    
    # 2. Gender (Radio Button)
    gender_map = st.sidebar.radio(
        "2. Gender", 
        options=[("Male", 0), ("Female", 1)], 
        format_func=lambda x: x[0] # Display 'Male'/'Female', but use 0/1 for prediction
    )[1]
    
    # 3. Monthly Spending (Number Input)
    monthly_spending = st.sidebar.number_input("3. Monthly Spending ($)", 10.0, 500.0, 250.0, step=5.0, help="Average monthly dollar value.")
    
    # 4. Subscription Length (Slider)
    subscription_length = st.sidebar.slider("4. Subscription Length (Months)", 1, 12, 6, help="How long the customer has been subscribed.")
    
    # 5. Support Interactions (Slider)
    support_interactions = st.sidebar.slider("5. Support Interactions (Last 6 Months)", 0, 5, 2, help="High interactions may indicate frustration.")
    
    st.sidebar.info("Prediction updates instantly as you adjust the inputs.")
    
    # ====================================================================
    # PREDICTION LOGIC
    # ====================================================================

    # Collect inputs into a DataFrame row (Maintains order)
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_map],
        'Monthly_Spending': [monthly_spending],
        'Subscription_Length': [subscription_length],
        'Support_Interactions': [support_interactions]
    })
    
    # Scale the input data using the *SAVED* fitted scaler
    input_scaled_values = scaler.transform(input_data[FEATURE_COLS])
    
    # Make prediction
    # model.predict_proba returns probability for [Class 0, Class 1]
    prediction_proba = model.predict_proba(input_scaled_values)[0]
    churn_proba = prediction_proba[1] # Probability of Churn (Class 1)

    # ====================================================================
    # MAIN PAGE OUTPUTS
    # ====================================================================
    st.write("---")
    st.subheader("🎯 Real-time Churn Risk Score")

    # Define risk levels for clear visualization
    if churn_proba >= 0.70:
        risk_level = "High Risk - IMMEDIATE ACTION REQUIRED"
        color = "red"
        icon = "🛑"
    elif churn_proba >= 0.40:
        risk_level = "Medium Risk - Proactive Offer Recommended"
        color = "orange"
        icon = "⚠️"
    else:
        risk_level = "Low Risk - Stable Customer"
        color = "green"
        icon = "✅"
    
    # Display the final prediction
    st.metric(
        label="Predicted Churn Probability", 
        value=f"{churn_proba:.2%}", # Display as percentage
        delta=f"Risk Level: {risk_level}",
        delta_color=color
    )

    st.markdown(f"**Retention Strategy:** {icon} <span style='color:{color}; font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
    
    # --- Actionable Insights ---
    st.write("---")
    st.subheader("🧠 Model Interpretation (Why This Prediction?)")
    
    # Provide simple, rule-based insights based on the input values
    if support_interactions >= 3:
        st.error("High risk factors detected: **Frequent Support Interactions**. Suggests high frustration or unresolved issues.")
    elif monthly_spending < 150:
        st.warning("Medium risk factor: **Low Monthly Spending**. Indicates low perceived value or light usage of the service.")
    else:
        st.success("Positive indicators: High spending and low interactions suggest a high-value, stable customer.")
        
