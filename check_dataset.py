import pandas as pd

file_path = "Dataset/train-00000-of-00001.parquet"

df = pd.read_parquet(file_path)

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nBasic Statistics:")
print(df.describe())

print("\nUnique States:", df["State"].nunique())
print("\nStates:")
print(df["State"].value_counts())

print("\nUnique Crops:", df["Crop"].nunique())
print("\nCrops:")
print(df["Crop"].value_counts())

print("\nUnique Seasons:", df["Season"].nunique())
print("\nSeasons:")
print(df["Season"].value_counts())

import matplotlib.pyplot as plt

crop_counts = df["Crop"].value_counts().head(15)

plt.figure(figsize=(10, 6))
crop_counts.sort_values().plot(kind="barh")

plt.title("Top 15 Crops by Number of Records")
plt.xlabel("Number of Records")
plt.ylabel("Crop")
plt.tight_layout()

plt.show()

crop_yield = (
    df.groupby("Crop")["Yield"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10, 6))
crop_yield.sort_values().plot(kind="barh")

plt.title("Top 15 Crops by Average Yield")
plt.xlabel("Average Yield")
plt.ylabel("Crop")
plt.tight_layout()

plt.show()

# Yield Distribution

plt.figure(figsize=(10, 6))
plt.hist(df["Yield"], bins=50)

plt.title("Distribution of Crop Yield")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.tight_layout()

plt.show()
# Yield Quantiles and Outlier Check

print("\nYield Quantiles:")
print(df["Yield"].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

Q1 = df["Yield"].quantile(0.25)
Q3 = df["Yield"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Yield"] < lower_bound) |
    (df["Yield"] > upper_bound)
]

print("\nIQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Number of Yield Outliers:", len(outliers))

# Outliers by Crop

outlier_crop_counts = (
    outliers["Crop"]
    .value_counts()
    .head(15)
)

print("\nTop 15 Crops Among Yield Outliers:")
print(outlier_crop_counts)

print("\nProduction / Area based Yield check:")

df["Calculated_Yield"] = df["Production"] / df["Area"]

print(df[["Production", "Area", "Yield", "Calculated_Yield"]].head(10))

print("\nCorrelation with Yield:")
print(
    df[
        ["Area", "Production", "Annual_Rainfall",
         "Fertilizer", "Pesticide", "Yield"]
    ].corr()["Yield"].sort_values(ascending=False)
)

print("\nActual Yield vs Production/Area:")

difference = (
    df["Yield"] - df["Calculated_Yield"]
).abs()

print("Mean Absolute Difference:", difference.mean())
print("Median Absolute Difference:", difference.median())
print("Maximum Difference:", difference.max())

print("\nSample comparison:")
print(
    df[
        ["Production", "Area", "Yield", "Calculated_Yield"]
    ].head(10)
)