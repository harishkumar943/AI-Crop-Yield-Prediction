import matplotlib.pyplot as plt


# Model performance from time-based evaluation
models = [
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting"
]

r2_scores = [
    -0.008588,
    0.973421,
    0.809769
]


# Create chart
plt.figure(figsize=(9, 6))

bars = plt.bar(models, r2_scores)

plt.title("Model Comparison - R² Score")
plt.xlabel("Machine Learning Model")
plt.ylabel("R² Score")

plt.ylim(-0.1, 1.1)

# Display values above bars
for bar, score in zip(bars, r2_scores):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        score + 0.02,
        f"{score:.3f}",
        ha="center"
    )


plt.tight_layout()

# Save chart
plt.savefig("model_comparison_r2.png", dpi=300)

plt.show()