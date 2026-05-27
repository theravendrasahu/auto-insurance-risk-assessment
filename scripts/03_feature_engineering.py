# ============================================================
# FILE: 03_feature_engineering.py
# PROJECT: Auto Insurance Risk Assessment using Telematics Data
# ============================================================

# ------------------------------------------------------------
# Import required libraries
# Pandas is used for data handling
# Matplotlib and Seaborn are used for visualization
# RandomForestClassifier is used to calculate feature importance
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# Load the preprocessed training datasets
# X_train contains input features
# y_train contains target labels
# ------------------------------------------------------------

X_train = pd.read_csv("X_train.csv")

y_train = pd.read_csv("y_train.csv")

# ------------------------------------------------------------
# Convert y_train into a one-dimensional array
# Some sklearn models require labels in 1D format
# ------------------------------------------------------------

y_train = y_train.values.ravel()

# ------------------------------------------------------------
# Display dataset shapes
# This helps verify successful loading of training data
# ------------------------------------------------------------

print("\n================ TRAINING DATA SHAPES ================\n")

print("X_train shape:", X_train.shape)

print("y_train shape:", y_train.shape)

# ------------------------------------------------------------
# Calculate correlation matrix for all features
# Correlation helps identify highly related features
# Very high correlation may create redundancy
# ------------------------------------------------------------

correlation_matrix = X_train.corr().abs()

# ------------------------------------------------------------
# Find feature pairs with correlation above 0.85
# Highly correlated features may negatively affect
# model interpretability and stability
# ------------------------------------------------------------

print("\n================ HIGH CORRELATION PAIRS ================\n")

high_corr_pairs = []

columns = correlation_matrix.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        
        corr_value = correlation_matrix.iloc[i, j]
        
        if corr_value > 0.85:
            
            feature_1 = columns[i]
            feature_2 = columns[j]
            
            high_corr_pairs.append((feature_1, feature_2, corr_value))
            
            print(
                f"{feature_1} <--> {feature_2} : "
                f"{corr_value:.2f}"
            )

# ------------------------------------------------------------
# Train a Random Forest model on the full training set
# Feature importance scores help identify which
# features contribute most to prediction performance
# ------------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# ------------------------------------------------------------
# Extract feature importance scores
# Higher importance means stronger contribution
# to the prediction process
# ------------------------------------------------------------

feature_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

# ------------------------------------------------------------
# Sort features from most important to least important
# ------------------------------------------------------------

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

# ------------------------------------------------------------
# Drop highly correlated features with lower importance
# If two features are strongly correlated,
# keep the more important feature and drop the weaker one
# ------------------------------------------------------------

features_to_drop = []

for feature_1, feature_2, corr_value in high_corr_pairs:
    
    importance_1 = feature_importance_df.loc[
        feature_importance_df["Feature"] == feature_1,
        "Importance"
    ].values[0]
    
    importance_2 = feature_importance_df.loc[
        feature_importance_df["Feature"] == feature_2,
        "Importance"
    ].values[0]
    
    if importance_1 < importance_2:
        features_to_drop.append(feature_1)
    else:
        features_to_drop.append(feature_2)

# Remove duplicate feature names
features_to_drop = list(set(features_to_drop))

# ------------------------------------------------------------
# Drop selected low-importance correlated features
# ------------------------------------------------------------

X_train = X_train.drop(columns=features_to_drop)

# ------------------------------------------------------------
# Save notes about dropped features
# This improves transparency and dissertation reporting
# ------------------------------------------------------------

with open("dropped_features.txt", "w") as file:
    
    if len(features_to_drop) == 0:
        file.write("No highly correlated features were dropped.\n")
    else:
        file.write("Dropped highly correlated features:\n")
        
        for feature in features_to_drop:
            file.write(f"- {feature}\n")

# ------------------------------------------------------------
# Retrain Random Forest model after feature removal
# This recalculates feature importance using the
# final optimized feature set
# ------------------------------------------------------------

rf_model.fit(X_train, y_train)

# ------------------------------------------------------------
# Extract updated feature importance scores
# ------------------------------------------------------------

updated_feature_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

# ------------------------------------------------------------
# Sort updated importance scores
# ------------------------------------------------------------

updated_feature_importance_df = (
    updated_feature_importance_df.sort_values(
        by="Importance",
        ascending=False
    )
)

# ------------------------------------------------------------
# Print top 10 most important features
# These are the strongest predictors for risk classification
# ------------------------------------------------------------

print("\n================ TOP 10 IMPORTANT FEATURES ================\n")

top_10_features = updated_feature_importance_df.head(10)

for index, row in top_10_features.iterrows():
    
    print(
        f"{row['Feature']} --> "
        f"{row['Importance']:.4f}"
    )

# ------------------------------------------------------------
# Create horizontal bar chart of feature importance
# The chart visually compares importance scores
# across all selected features
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))

sns.barplot(
    data=updated_feature_importance_df,
    x="Importance",
    y="Feature",
    palette="viridis"
)

plt.title(
    "Feature Importance from Random Forest",
    fontsize=16
)

plt.xlabel(
    "Importance Score",
    fontsize=12
)

plt.ylabel(
    "Feature Name",
    fontsize=12
)

plt.tight_layout()

# ------------------------------------------------------------
# Save the feature importance chart as PNG
# ------------------------------------------------------------

plt.savefig("feature_importance.png")

plt.close()

# ------------------------------------------------------------
# Save the final processed feature dataset
# This dataset can be reused during model training
# ------------------------------------------------------------

X_train.to_csv(
    "X_train_feature_selected.csv",
    index=False
)

# ------------------------------------------------------------
# Final completion messages
# ------------------------------------------------------------

print("\nFeature engineering completed successfully.")

print("\nSaved files:")
print("1. feature_importance.png")
print("2. dropped_features.txt")
print("3. X_train_feature_selected.csv")