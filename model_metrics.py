import pandas as pd

metrics = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        92.963397,
        12.918365,
        36.387807
    ],
    "RMSE": [
        989.489690,
        160.629142,
        429.728901
    ],
    "R2": [
        -0.008588,
        0.973421,
        0.809769
    ]
})

metrics.to_csv(
    "model_metrics.csv",
    index=False
)

print("Model metrics saved successfully!")
print(metrics)