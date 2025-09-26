import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

output_dir = "models"
os.makedirs(output_dir, exist_ok=True)


X_train = pd.read_csv("data/normalized/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

model = RandomForestRegressor(random_state=42)

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

# GridSearch
grid = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1)
grid.fit(X_train.drop(columns=["date"], errors="ignore"), y_train)

# Save best parameters
best_params_path = os.path.join(output_dir, "best_params.pkl")
with open(best_params_path, "wb") as f:
    pickle.dump(grid.best_params_, f)

print("Best parameters found:", grid.best_params_)
print("Saved to:", best_params_path)
