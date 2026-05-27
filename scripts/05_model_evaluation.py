  # ============================================================
# FILE: 05_model_evaluation.py
# PROJECT: Auto Insurance Risk Assessment using Telematics Data
# ============================================================

# ------------------------------------------------------------
# Import required libraries
# Pandas is used for data handling
# Matplotlib and Seaborn are used for visualization
# sklearn metrics are used for model evaluation
# joblib is used for loading saved models
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

# ------------------------------------------------------------
# Load testing datasets
# X_test contains feature values
# y_test contains actual target labels
# ------------------------------------------------------------

X_test = pd.read_csv("X_test.csv")

y_test = pd.read_csv("y_test.csv")

# ------------------------------------------------------------
# Convert y_test into one-dimensional format
# Some sklearn functions require 1D labels
# ------------------------------------------------------------

y_test = y_test.values.ravel()

# ------------------------------------------------------------
# Load training datasets
# Training data is required to retrain three models
# ------------------------------------------------------------

X_train = pd.read_csv("X_train.csv")

y_train = pd.read_csv("y_train.csv")

y_train = y_train.values.ravel()

# ------------------------------------------------------------
# Load the previously saved best model
# This model achieved the highest F1 score earlier
# ------------------------------------------------------------

best_model = joblib.load("best_model.pkl")

# ------------------------------------------------------------
# Recreate and retrain all four models
# These settings match the training configuration
# from script 04_model_training.py
# ------------------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
    )
}

# ------------------------------------------------------------
# Train all models on the training dataset
# ------------------------------------------------------------

for model_name, model in models.items():

    model.fit(X_train, y_train)

# ------------------------------------------------------------
# Replace the best model object with loaded model
# This ensures evaluation uses the saved final model
# ------------------------------------------------------------

best_model_name = type(best_model).__name__

if "LogisticRegression" in best_model_name:
    models["Logistic Regression"] = best_model

elif "DecisionTreeClassifier" in best_model_name:
    models["Decision Tree"] = best_model

elif "RandomForestClassifier" in best_model_name:
    models["Random Forest"] = best_model

elif "XGBClassifier" in best_model_name:
    models["XGBoost"] = best_model

# ------------------------------------------------------------
# Convert multiclass labels into binary format
# ROC curves require binarized class labels
# ------------------------------------------------------------

classes = [0, 1, 2]

y_test_bin = label_binarize(
    y_test,
    classes=classes
)

# ------------------------------------------------------------
# Create ROC curve comparison plot
# ROC curves compare classification performance
# across different probability thresholds
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))

# Define colors for each model
model_colors = {
    "Logistic Regression": "blue",
    "Decision Tree": "green",
    "Random Forest": "orange",
    "XGBoost": "red"
}

# ------------------------------------------------------------
# Store model evaluation metrics for summary chart
# ------------------------------------------------------------

evaluation_results = []

# ------------------------------------------------------------
# Evaluate all models and generate ROC curves
# ------------------------------------------------------------

for model_name, model in models.items():

    # --------------------------------------------------------
    # Predict probabilities for multiclass ROC analysis
    # --------------------------------------------------------

    y_prob = model.predict_proba(X_test)

    # --------------------------------------------------------
    # Calculate micro-average ROC curve and AUC score
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test_bin.ravel(),
        y_prob.ravel()
    )

    roc_auc = auc(fpr, tpr)

    # --------------------------------------------------------
    # Plot ROC curve for current model
    # --------------------------------------------------------

    plt.plot(
        fpr,
        tpr,
        color=model_colors[model_name],
        linewidth=2,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )

    # --------------------------------------------------------
    # Predict class labels for metric calculations
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    # --------------------------------------------------------
    # Save evaluation results for comparison chart
    # --------------------------------------------------------

    evaluation_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# ------------------------------------------------------------
# Add diagonal baseline line
# ------------------------------------------------------------

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black"
)

# ------------------------------------------------------------
# Customize ROC chart
# ------------------------------------------------------------

plt.title(
    "ROC Curve Comparison for ML Models",
    fontsize=16
)

plt.xlabel(
    "False Positive Rate",
    fontsize=12
)

plt.ylabel(
    "True Positive Rate",
    fontsize=12
)

plt.legend(loc="lower right")

plt.tight_layout()

# ------------------------------------------------------------
# Save ROC curve comparison chart
# ------------------------------------------------------------

plt.savefig("roc_curves_comparison.png")

plt.close()

# ------------------------------------------------------------
# Convert evaluation results into DataFrame
# This creates a structured comparison table
# ------------------------------------------------------------

results_df = pd.DataFrame(evaluation_results)

# ------------------------------------------------------------
# Print evaluation summary table
# ------------------------------------------------------------

print("\n================ MODEL EVALUATION SUMMARY ================\n")

print(results_df)

# ------------------------------------------------------------
# Create grouped bar chart for metric comparison
# This chart visually compares all models
# across multiple evaluation metrics
# ------------------------------------------------------------

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

x = np.arange(len(results_df["Model"]))

width = 0.2

plt.figure(figsize=(12, 7))

# ------------------------------------------------------------
# Plot grouped bars for each evaluation metric
# ------------------------------------------------------------

for i, metric in enumerate(metrics):

    plt.bar(
        x + i * width,
        results_df[metric],
        width=width,
        label=metric
    )

# ------------------------------------------------------------
# Customize grouped bar chart
# ------------------------------------------------------------

plt.xticks(
    x + width * 1.5,
    results_df["Model"]
)

plt.ylabel(
    "Metric Score",
    fontsize=12
)

plt.xlabel(
    "Machine Learning Models",
    fontsize=12
)

plt.title(
    "Model Performance Comparison",
    fontsize=16
)

plt.legend()

plt.tight_layout()

# ------------------------------------------------------------
# Save grouped metrics comparison chart
# ------------------------------------------------------------

plt.savefig("model_metrics_comparison.png")

plt.close()

# ------------------------------------------------------------
# Identify best performing model using F1 score
# F1 score balances precision and recall
# ------------------------------------------------------------

best_model_row = results_df.loc[
    results_df["F1 Score"].idxmax()
]

# ------------------------------------------------------------
# Print final recommendation statement
# ------------------------------------------------------------

print("\n================ FINAL RECOMMENDATION ================\n")

print(
    f"The recommended model for the Auto Insurance "
    f"Risk Assessment System is "
    f"{best_model_row['Model']}."
)

print(
    f"This model achieved the highest F1 Score of "
    f"{best_model_row['F1 Score']:.4f}, "
    f"along with strong Accuracy, Precision, "
    f"and Recall performance."
)

print(
    "The model provides the best balance between "
    "prediction accuracy and reliable risk classification "
    "for telematics based insurance analysis."
)

# ------------------------------------------------------------
# Final completion message
# ------------------------------------------------------------

print("\nModel evaluation completed successfully.")

print("\nSaved files:")
print("1. roc_curves_comparison.png")
print("2. model_metrics_comparison.png")