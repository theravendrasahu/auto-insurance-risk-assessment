# ============================================================
# FILE: 04_model_training.py
# PROJECT: Auto Insurance Risk Assessment using Telematics Data
# ============================================================

# ------------------------------------------------------------
# Import required libraries
# Pandas is used for handling datasets and summary tables
# Matplotlib and Seaborn are used for visualizations
# sklearn models and metrics are used for ML training
# joblib is used to save the best trained model
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ------------------------------------------------------------
# Load training and testing datasets
# X files contain features
# y files contain target labels
# ------------------------------------------------------------

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# ------------------------------------------------------------
# Convert target labels into one-dimensional arrays
# Some ML models require labels in 1D format
# ------------------------------------------------------------

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# ------------------------------------------------------------
# Display dataset shapes
# This helps verify successful dataset loading
# ------------------------------------------------------------

print("\n================ DATASET SHAPES ================\n")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# ------------------------------------------------------------
# Define all machine learning models
# Each model provides different advantages
# ------------------------------------------------------------

models = {

    "Logistic_Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision_Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "Random_Forest": RandomForestClassifier(
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
# Create an empty list to store model evaluation results
# This list will later become a summary comparison table
# ------------------------------------------------------------

model_results = []

# ------------------------------------------------------------
# Train and evaluate each machine learning model
# ------------------------------------------------------------

for model_name, model in models.items():

    print("\n================================================")
    print(f"TRAINING MODEL: {model_name}")
    print("================================================\n")

    # --------------------------------------------------------
    # Train the model using training data
    # --------------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predict risk labels on the test dataset
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # Calculate evaluation metrics
    # Accuracy measures overall correctness
    # Precision measures prediction quality
    # Recall measures detection ability
    # F1 Score balances precision and recall
    # --------------------------------------------------------

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
    # Print classification report
    # This gives detailed class-wise evaluation metrics
    # --------------------------------------------------------

    print("Accuracy:", round(accuracy, 4))

    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred))

    # --------------------------------------------------------
    # Store evaluation metrics for final comparison table
    # --------------------------------------------------------

    model_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    # --------------------------------------------------------
    # Generate confusion matrix
    # Confusion matrix shows prediction performance
    # across all risk categories
    # --------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"]
    )

    plt.title(
        f"Confusion Matrix - {model_name}",
        fontsize=14
    )

    plt.xlabel(
        "Predicted Label",
        fontsize=12
    )

    plt.ylabel(
        "Actual Label",
        fontsize=12
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save confusion matrix plot as PNG image
    # --------------------------------------------------------

    plt.savefig(
        f"confusion_matrix_{model_name}.png"
    )

    plt.close()

# ------------------------------------------------------------
# Create a summary comparison table using Pandas
# This table compares performance of all models
# ------------------------------------------------------------

results_df = pd.DataFrame(model_results)

# ------------------------------------------------------------
# Sort models by F1 score from highest to lowest
# F1 score is selected because it balances
# precision and recall together
# ------------------------------------------------------------

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

# ------------------------------------------------------------
# Print final model comparison table
# ------------------------------------------------------------

print("\n================ MODEL COMPARISON TABLE ================\n")

print(results_df)

# ------------------------------------------------------------
# Identify the best performing model
# The model with highest F1 score is selected
# ------------------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_f1_score = results_df.iloc[0]["F1_Score"]

best_model = models[best_model_name]

# ------------------------------------------------------------
# Save the best model using joblib
# This model can later be reused for deployment
# or prediction on new driver data
# ------------------------------------------------------------

joblib.dump(best_model, "best_model.pkl")

# ------------------------------------------------------------
# Print final best model selection result
# ------------------------------------------------------------

print("\n================ BEST MODEL SELECTED ================\n")

print(f"Best Model: {best_model_name}")

print(f"Best F1 Score: {round(best_f1_score, 4)}")

print(
    f"\n{best_model_name} was saved as best_model.pkl "
    f"because it achieved the highest F1 score."
)

# ------------------------------------------------------------
# Final completion message
# ------------------------------------------------------------

print("\nModel training and evaluation completed successfully.")

print("\nSaved files:")

for model_name in models.keys():

    print(f"- confusion_matrix_{model_name}.png")

print("- best_model.pkl")