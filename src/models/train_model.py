# src/models/train_model.py

import pandas as pd
import numpy as np
from sklearn import ensemble
import joblib
import os

# Load training and test sets
X_train = pd.read_csv("data/normalized/X_train_scaled.csv")
X_test = pd.read_csv("data/normalized/X_test_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")

# Flatten y arrays
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Try to load best parameters from GridSearch
best_params_path = "models/best_params.pkl"
if os.path.exists(best_params_path):
    best_params = joblib.load(best_params_path)
    print("Loaded best parameters:", best_params)
    rf_regressor = ensemble.RandomForestRegressor(
        **best_params, random_state=42, n_jobs=-1
    )
else:
    print("No best_params.pkl found. Using default parameters.")
    rf_regressor = ensemble.RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )

# Train the model
rf_regressor.fit(X_train, y_train)

# Save the trained model
os.makedirs("models", exist_ok=True)
model_filename = "models/trained_model.joblib"
joblib.dump(rf_regressor, model_filename)

print("Model trained and saved successfully.")
