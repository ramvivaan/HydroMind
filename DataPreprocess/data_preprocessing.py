import pandas as pd
import os
import numpy as np


BASE_DIR =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE_DIR, "Dataset", "Data.csv")
output_path = os.path.join(BASE_DIR, "Dataset", "master_data.csv")
#Dropping unneccesary columns
df = pd.read_csv(path, encoding = "latin1")
df = df.drop("Number", axis = 1)
df = df.drop("Station", axis = 1)
df = df.drop("Zone", axis = 1)
df = df.drop("Time stamp (Sampling)", axis = 1)
df = df.drop("Catchment", axis = 1)
df = df.drop("Nitrogen - total / WCK - Simultaneous Determination of Total Nitrogen and Total Phosphorus using Persulphate Digestion [mg/L]", axis = 1)
df = df.drop("Nitrate + nitrite as N (NOx) / WCK - Determination of Oxidised Nitrogen Nitrate & Nitrite in Water [mg/L]", axis = 1)
df = df.drop("Phosphorus - reactive (orthophosphate) - dissolved / WCK - Determination of Reactive Phosphorus [mg/L]", axis = 1)
df = df.drop("Solids - total suspended @ 105 C / WCK - Total Suspended Solids at 105°C [mg/L]", axis = 1)
df =  df.drop(r"Oxygen - dissolved saturation / Field - Water quality sonde [% sat]", axis = 1)
df = df.drop("Phosphorus - total / WCK - Simultaneous Determination of Total Nitrogen and Total Phosphorus using Persulphate Digestion [mg/L]", axis = 1)



#renaming columns
df.rename(columns= {"Electrical Conductivity @25 C / WCK - Determination of Conductivity in Water [µS/cm]":"EC", "Turbidity / Field - Determination of turbidity by nephelometry using HACH turbidimeter [NTU]":"Turbidity", "pH / Field - Water quality sonde [---]" : "pH"}, inplace=True)



#deleting rows with missing values
df = df.dropna(axis = 0)


#adding label column for labels
def label_row(row):
    #defining pH score
    max_ph_deviation = 1.5
    ph_deviation = abs(7-row["pH"])
    if ph_deviation > max_ph_deviation:
        ph_score = 0

    else:
        ph_score = (1 - ph_deviation/ max_ph_deviation)

    # defning EC score
    max_EC = 3000
    ideal_EC = 1000
    if row["EC"] > max_EC:
        EC_score = 0


    elif row["EC"] <= ideal_EC:
        EC_score = 1


    else:
        EC_score = 1 - ((row["EC"] - ideal_EC)/(max_EC - ideal_EC))

    #defining turbidity score
    ideal_turbidity = 0
    max_turbidity = 5
    if row["Turbidity"] > max_turbidity:
        turbidity_score = 0
    elif row["Turbidity"] == ideal_turbidity:
        turbidity_score = 1
    else:
        turbidity_score = 1 - (row["Turbidity"] - ideal_turbidity)/(max_turbidity - ideal_turbidity)

 
    #defining how much each factor is depended on by overall score
    water_safety_score = 100 * (0.4 * float(ph_score) + 0.3 * float(EC_score) + 0.3 * float(turbidity_score))

    return water_safety_score
    

df["Safety Score"] = df.apply(label_row, axis = 1)









master_df = df.copy()

# pH noise
master_df["pH"] += np.random.normal(0, 0.1, size=len(master_df))

# EC noise (percentage-based)
master_df["EC"] += master_df["EC"] * np.random.normal(0, 0.02, size=len(master_df))

# Turbidity noise
master_df["Turbidity"] += master_df["Turbidity"] * np.random.normal(0, 0.08, size=len(master_df))

master_df["pH"] = master_df["pH"].clip(0, 14)
master_df["EC"] = master_df["EC"].clip(lower=0)
master_df["Turbidity"] = master_df["Turbidity"].clip(lower=0)




#delta = master_df["pH"] - df["pH"]
#print(delta.abs().describe())
