import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Input/output paths
metrics_dir = "metrics"
data_dir = "data"
os.makedirs(metrics_dir, exist_ok=True)

# Load test data
X_test = pd.read_csv("data/normalized/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")
y_test = np.ravel(y_test)

# Load trained model
model_path = "models/trained_model.joblib"
model = joblib.load(model_path)

# Predict
y_pred = model.predict(X_test)

# Save predictions
predictions = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})
predictions.to_csv(os.path.join(data_dir, "predictions.csv"), index=False)

# Compute metrics
scores = {
    "mse": float(mean_squared_error(y_test, y_pred)),
    "mae": float(mean_absolute_error(y_test, y_pred)),
    "r2": float(r2_score(y_test, y_pred))
}

# Save metrics
with open(os.path.join(metrics_dir, "scores.json"), "w") as f:
    json.dump(scores, f)

print("Evaluation complete. Predictions saved in data/processed_data/predictions.csv. Metrics saved in metrics/scores.json")
