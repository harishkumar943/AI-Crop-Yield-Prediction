import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
file_path = "Dataset/train-00000-of-00001.parquet"
df = pd.read_parquet(file_path)

print("Dataset Loaded:", df.shape)


# Features and target
X = df.drop("Yield", axis=1)
y = df["Yield"]


# Categorical and numerical columns
categorical_columns = ["State", "Crop", "Season"]
numerical_columns = [
    "Year",
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]


# Preprocessing
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


# Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# Complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))


# Train model
pipeline.fit(X_train, y_train)


# Predictions
y_pred = pipeline.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print("\nModel Performance")
print("-----------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)