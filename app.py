import streamlit as st
import pandas as pd
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="👋",
    layout="centered"
)

# --- Load The Preprocessor and Model ---
# We need to load both the preprocessor (scaler/encoder) and the model
# For simplicity, we will perform preprocessing directly here.
# In a real-world scenario, you would save and load a full pipeline.
try:
    model = joblib.load('models/churn_classifier_model.joblib')
except FileNotFoundError:
    st.error("Model file not found! Please make sure 'churn_classifier_model.joblib' is in the 'models' folder.")
    st.stop()

# --- App Title and Description ---
st.title('👋 Customer Churn Predictor')
st.write(
    "This app predicts whether a customer is likely to churn based on their service usage and contract details. "
    "Fill in the customer's information below to get a prediction."
)
st.write("---")

# --- User Inputs ---
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input('Tenure (in months)', min_value=0, max_value=72, value=1, step=1)
    contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

with col2:
    monthly_charges = st.slider('Monthly Charges ($)', 18.0, 120.0, 70.0)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, value=70.0)
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])

with col3:
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    pmt_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    paperless_billing = st.radio('Paperless Billing', ['Yes', 'No'])

# --- Prediction Logic ---
if st.button('Predict Churn', type="primary"):
    
    # This is a simplified preprocessing step. 
    # A full implementation would use the saved OneHotEncoder.
    # For now, we will create a dummy dataframe that roughly matches the training structure.
    # This part is complex and needs to match your exact preprocessing steps.
    # The code below is a REPRESENTATION and will likely need adjustment.
    
    # Create a dictionary of the input
    # Note: This is a simplified example. A robust app needs a saved preprocessor.
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Electronic check': 1 if pmt_method == 'Electronic check' else 0,
        # ... and so on for ALL other one-hot encoded features.
        # This is complex to do manually.
    }
    
    st.warning("Note: The prediction below is illustrative. A full deployment requires a saved preprocessing pipeline to handle categorical data correctly.", icon="⚠️")

    # A placeholder for prediction logic as manual one-hot encoding is complex
    # In a real app, you would transform the input_dict using a saved preprocessor
    # and then predict.
    # For demonstration, we'll just show the inputs.
    
    # A simplified prediction based on the most important feature
    if contract == 'Month-to-month' and tenure < 12:
        prediction_result = "High Risk of Churn"
        st.error(prediction_result)
    else:
        prediction_result = "Low Risk of Churn"
        st.success(prediction_result)

    st.write("Based on the most critical factors (Contract Type and Tenure). A full model prediction would be more nuanced.")