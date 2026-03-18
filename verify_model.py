import pickle
import numpy as np
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

try:
    print("Loading artifacts...")
    model_path = os.path.join(script_dir, 'rf_model.pkl')
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Artifacts loaded successfully.")

    # Create dummy input data (based on diabetes.csv schema)
    # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    # Example values: 1, 85, 66, 29, 0, 26.6, 0.351, 31 (Outcome 0)
    input_data = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
    
    print("Scaling input...")
    input_scaled = scaler.transform(input_data)
    
    print("Predicting...")
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)
    
    print(f"Prediction: {prediction[0]}")
    print(f"Probability: {prob[0]}")
    print("Verification passed!")

except Exception as e:
    print(f"Verification FAILED: {e}")
