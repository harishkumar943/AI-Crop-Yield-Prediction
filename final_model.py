import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

file_path = "Dataset/train-00000-of-00001.parquet"

df = pd.read_parquet(file_path)

print("Dataset Loaded:", df.shape)


# --------------------------------------------------
# Features and Target
# --------------------------------------------------

# Production is removed to avoid target leakage
X = df.drop(["Yield", "Production"], axis=1)

# Log transformation of target
y = np.log1p(df["Yield"])


# --------------------------------------------------
# Columns
# --------------------------------------------------

categorical_columns = [
    "State",
    "Crop",
    "Season"
]

numerical_columns = [
    "Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# Random Forest Model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# Complete Pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# Train Final Model
# --------------------------------------------------

print("\nTraining Final Random Forest Model...")

pipeline.fit(X, y)


# --------------------------------------------------
# Save Model
# --------------------------------------------------

model_path = "crop_yield_model.pkl"

joblib.dump(pipeline, model_path)


print("\nFinal model trained successfully!")
print("Model saved as:", model_path)