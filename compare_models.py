import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

file_path = "Dataset/train-00000-of-00001.parquet"

df = pd.read_parquet(file_path)

print("Dataset Loaded:", df.shape)


# --------------------------------------------------
# Time-based train-test split
# --------------------------------------------------

train_df = df[df["Year"] <= 2023].copy()
test_df = df[df["Year"] >= 2024].copy()

print("\nTraining Years: 2000-2023")
print("Testing Years: 2024-2026")

print("Training Records:", len(train_df))
print("Testing Records:", len(test_df))


# --------------------------------------------------
# Features and Target
# --------------------------------------------------

X_train = train_df.drop(
    ["Yield", "Production"],
    axis=1
)

X_test = test_df.drop(
    ["Yield", "Production"],
    axis=1
)

y_train = np.log1p(train_df["Yield"])
y_test = np.log1p(test_df["Yield"])


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
# Models
# --------------------------------------------------

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}


# --------------------------------------------------
# Model comparison
# --------------------------------------------------

results = []

random_forest_predictions = None


for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    # Prediction on unseen 2024-2026 data
    y_pred_log = pipeline.predict(X_test)

    # Convert predictions back to original Yield scale
    y_pred = np.expm1(y_pred_log)

    # Convert actual values back to original scale
    y_test_original = np.expm1(y_test)

    mae = mean_absolute_error(
        y_test_original,
        y_pred
    )

    rmse = mean_squared_error(
        y_test_original,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_test_original,
        y_pred
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # Save Random Forest test predictions
    if name == "Random Forest":

        random_forest_predictions = pd.DataFrame({
            "Actual_Yield": y_test_original.values,
            "Predicted_Yield": y_pred
        })


# --------------------------------------------------
# Save Random Forest predictions
# --------------------------------------------------

random_forest_predictions.to_csv(
    "random_forest_test_predictions.csv",
    index=False
)

print("\nRandom Forest test predictions saved as:")
print("random_forest_test_predictions.csv")


# --------------------------------------------------
# Display results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n================ TIME-BASED MODEL COMPARISON ================")

print(
    results_df.to_string(index=False)
)

print("==============================================================")