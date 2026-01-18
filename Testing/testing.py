import joblib
import os
import pandas as pd


BASE_DIR =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE_DIR, "Dataset", "master_data.csv")
model_path = os.path.join(BASE_DIR, "models", "decision_tree_model.pkl")
data_path = os.path.join(BASE_DIR, "Dataset", "master_data.csv")



loaded_model = joblib.load(model_path)
df = pd.read_csv(data_path, encoding = "latin1")


for i in range(0,10):
    row = df.iloc[i]
    input_data = [[row["EC"], row["Turbidity"], row["pH"]]]

    predicted_score = loaded_model.predict(input_data)

    print(f"Actual Safety Score for row {i}:", row["Safety Score"])
    print(f"Predicted Safety Score for row {i}:", predicted_score[0])
    print(f"Difference for row {i} = ", abs(row["Safety Score"] - predicted_score[0]))
    i+=1