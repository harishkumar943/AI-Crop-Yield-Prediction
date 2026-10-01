import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Load Random Forest test predictions
# --------------------------------------------------

file_path = "random_forest_test_predictions.csv"

df = pd.read_csv(file_path)

print("Prediction Data Loaded:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# --------------------------------------------------
# Actual vs Predicted Yield
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    df["Actual_Yield"],
    df["Predicted_Yield"],
    alpha=0.5
)

# Reference line: Perfect Prediction
minimum = min(
    df["Actual_Yield"].min(),
    df["Predicted_Yield"].min()
)

maximum = max(
    df["Actual_Yield"].max(),
    df["Predicted_Yield"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted Crop Yield")
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted_yield.png",
    dpi=300
)

plt.show()

print("\nGraph saved as:")
print("actual_vs_predicted_yield.png")