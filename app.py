import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and scaler
@st.cache_resource
def load_artifacts():
    model_path = os.path.join(script_dir, 'rf_model.pkl')
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please run train_model.py first.")
    st.stop()

# Streamlit App
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction Model")
st.markdown("Enter the patient's details below to predict the likelihood of diabetes.")

# Input Form
with st.form("prediction_form"):
    st.header("Patient Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1, help="Number of times pregnant")
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70, help="Diastolic blood pressure (mm Hg)")
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness (mm)")

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, help="2-Hour serum insulin (mu U/ml)")
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, help="Body mass index (weight in kg/(height in m)^2)")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f", help="Diabetes pedigree function")
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

    submit_button = st.form_submit_button("Predict Risk")

if submit_button:
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    st.divider()
    
    if prediction[0] == 1:
        st.error(f"**High Risk Detected**")
        st.write(f"The model predicts a high probability ({prediction_proba[0][1]*100:.2f}%) of diabetes.")
    else:
        st.success(f"**Low Risk Detected**")
        st.write(f"The model predicts a low probability ({prediction_proba[0][1]*100:.2f}%) of diabetes.")

st.markdown("---")
st.caption("Note: This is a machine learning model and should not be used as a substitute for professional medical advice.")

if __name__ == "__main__":
    print("\n\n\033[93mWARNING: This script should be run with Streamlit, not Python directly.\033[0m")
    print("Please use the following command to run the app:")
    print("\n    \033[92mstreamlit run app.py\033[0m\n")
