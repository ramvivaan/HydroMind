from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt

BASE_DIR =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE_DIR, "Dataset", "master_data.csv")
model_path = os.path.join(BASE_DIR, "models", "decision_tree_model.pkl")



df = pd.read_csv(path, encoding = "latin1")


x = df[["EC", "Turbidity", "pH"]]
y = df["Safety Score"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)



tree_model  = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state= 42)


tree_model.fit(x_train, y_train)


y_pred = tree_model.predict(x_test)

joblib.dump(tree_model, model_path)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
