import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import os

# Load dataset
# Use absolute path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'diabetes.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"The dataset file was not found at: {data_path}. Please ensure 'diabetes.csv' is in the same directory as this script.")

data = pd.read_csv(data_path)

# Feature selection (using all features as per notebook analysis)
X = data.drop(['Outcome'], axis=1)
y = data['Outcome']

# Preprocessing
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Splitting dataset (optional for final model, but good for verification)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

# Retrain on full dataset for the app
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_scaled, y)

# Save model and scaler
model_path = os.path.join(script_dir, 'rf_model.pkl')
scaler_path = os.path.join(script_dir, 'scaler.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
