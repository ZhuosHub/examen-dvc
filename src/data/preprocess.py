import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Input and output paths
input_dir = "data/processed_data"
output_dir = "data/normalized"
os.makedirs(output_dir, exist_ok=True)

# Load training and test sets
X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))

# Initialize scaler
scaler = StandardScaler()

# Fit on training numeric columns
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame with original column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save scaled datasets
X_train_scaled.to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False)
X_test_scaled.to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False)

print("Scaling complete. Scaled datasets saved in", output_dir)
