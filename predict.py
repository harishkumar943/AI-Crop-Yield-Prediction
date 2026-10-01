import pandas as pd
import joblib


# Load trained model
model = joblib.load("crop_yield_model.pkl")

print("Model loaded successfully!")


# New crop data
new_data = pd.DataFrame([{
    "Year": 2026,
    "State": "Uttar Pradesh",
    "Crop": "Wheat",
    "Season": "Rabi",
    "Area": 1000,
    "Production": 2000,
    "Annual_Rainfall": 800,
    "Fertilizer": 50000,
    "Pesticide": 1000
}])


# Predict
prediction_log = model.predict(new_data)

# Convert prediction back to original Yield scale
prediction = prediction_log[0]

import numpy as np

prediction = np.expm1(prediction)


print("\n========== CROP YIELD PREDICTION ==========")
print("State:", new_data["State"].iloc[0])
print("Crop:", new_data["Crop"].iloc[0])
print("Season:", new_data["Season"].iloc[0])
print("Predicted Yield:", prediction)
print("===========================================")