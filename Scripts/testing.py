import joblib
import os
import pandas as pd
from training_code import r2, r2_score, y_pred, y_test, rmse
from matplotlib import pyplot as plt

BASE_DIR =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE_DIR, "Dataset", "master_data.csv")
model_path = os.path.join(BASE_DIR, "models", "decision_tree_model.pkl")
data_path = os.path.join(BASE_DIR, "Dataset", "master_data.csv")



loaded_model = joblib.load(model_path)
df = pd.read_csv(data_path, encoding = "latin1")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([0, 100], [0, 100], color='red', linestyle='--')  # perfect prediction line
plt.xlabel("Actual Safety Score")
plt.ylabel("Predicted Safety Score")
plt.title("Predicted vs Actual Safety Scores")
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, color='purple', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Safety Score")
plt.ylabel("Residual (Error)")
plt.title("Residuals Plot")
plt.show()
