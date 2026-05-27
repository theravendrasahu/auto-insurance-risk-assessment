# ============================================================
# FILE: 02_preprocessing.py
# PROJECT: Auto Insurance Risk Assessment using Telematics Data
# ============================================================

# ------------------------------------------------------------
# Import required libraries
# Pandas is used for data handling
# NumPy is used for numerical operations
# sklearn modules are used for preprocessing and data splitting
# joblib is used to save preprocessing objects
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Load the dataset from CSV file
# The dataset contains driver, vehicle, telematics,
# connectivity, and risk classification data
# ------------------------------------------------------------

df = pd.read_csv("telematics_sample_dataset.csv")

# ------------------------------------------------------------
# Display initial dataset shape
# This helps verify successful loading of the dataset
# ------------------------------------------------------------

print("\n================ ORIGINAL DATASET SHAPE ================\n")
print(df.shape)

# ------------------------------------------------------------
# Drop columns that should not be used as input features
# driver_id and vehicle_id are identifiers only
# risk_score and risk_category are output-related columns
# ------------------------------------------------------------

df = df.drop(
    columns=[
        "driver_id",
        "vehicle_id",
        "risk_score",
        "risk_category"
    ]
)

# ------------------------------------------------------------
# Handle missing values in numerical columns
# Numerical missing values are filled using the median
# Median is less sensitive to outliers than mean
# ------------------------------------------------------------

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

# ------------------------------------------------------------
# Handle missing values in categorical columns
# Categorical missing values are filled using the mode
# Mode represents the most frequent category
# ------------------------------------------------------------

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# ------------------------------------------------------------
# Apply Label Encoding to vehicle_type column
# Machine learning models require numerical inputs
# LabelEncoder converts text categories into numbers
# ------------------------------------------------------------

encoder = LabelEncoder()

df["vehicle_type"] = encoder.fit_transform(df["vehicle_type"])

# ------------------------------------------------------------
# Apply completeness override rule
# Any driver with trip_completeness_pct below 60
# is automatically classified as High Risk (label = 2)
# This prevents selective disconnection fraud
# ------------------------------------------------------------

df.loc[
    df["trip_completeness_pct"] < 60,
    "risk_label"
] = 2

# ------------------------------------------------------------
# Separate features (X) and target labels (y)
# risk_label is the prediction target variable
# ------------------------------------------------------------

X = df.drop(columns=["risk_label"])

y = df["risk_label"]

# ------------------------------------------------------------
# Identify numerical feature columns for scaling
# StandardScaler standardizes feature values
# This improves ML model performance and stability
# ------------------------------------------------------------

numerical_feature_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

# ------------------------------------------------------------
# Apply feature scaling using StandardScaler
# Features are transformed to mean = 0 and std = 1
# ------------------------------------------------------------

scaler = StandardScaler()

X[numerical_feature_columns] = scaler.fit_transform(
    X[numerical_feature_columns]
)

# ------------------------------------------------------------
# Split the dataset into training and testing sets
# 80% data used for training and 20% for testing
# Stratified split preserves class distribution
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ------------------------------------------------------------
# Save processed training and testing datasets as CSV files
# These files will be used during model training
# ------------------------------------------------------------

X_train.to_csv("X_train.csv", index=False)

X_test.to_csv("X_test.csv", index=False)

y_train.to_csv("y_train.csv", index=False)

y_test.to_csv("y_test.csv", index=False)

# ------------------------------------------------------------
# Save fitted preprocessing objects using joblib
# The saved scaler and encoder will be reused later
# during prediction and deployment
# ------------------------------------------------------------

joblib.dump(scaler, "scaler.pkl")

joblib.dump(encoder, "encoder.pkl")

# ------------------------------------------------------------
# Print shapes of saved datasets
# This helps verify preprocessing and splitting results
# ------------------------------------------------------------

print("\n================ SAVED FILE SHAPES ================\n")

print("X_train shape:", X_train.shape)

print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)

print("y_test shape:", y_test.shape)

# ------------------------------------------------------------
# Final completion message
# ------------------------------------------------------------

print("\nPreprocessing completed successfully.")

print("\nSaved files:")
print("1. X_train.csv")
print("2. y_train.csv")
print("3. X_test.csv")
print("4. y_test.csv")
print("5. scaler.pkl")
print("6. encoder.pkl")