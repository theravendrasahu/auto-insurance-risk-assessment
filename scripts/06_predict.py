# ============================================================
# FILE: 06_predict.py
# PROJECT: Auto Insurance Risk Assessment using Telematics Data
# ============================================================

# ------------------------------------------------------------
# Import required libraries
# Pandas is used for creating structured input data
# NumPy is used for numerical operations
# joblib is used for loading saved ML objects
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# Load saved preprocessing objects and trained model
# scaler.pkl standardizes numerical features
# encoder.pkl converts vehicle_type into numeric form
# best_model.pkl contains the final trained ML model
# ------------------------------------------------------------

scaler = joblib.load("scaler.pkl")

encoder = joblib.load("encoder.pkl")

model = joblib.load("best_model.pkl")

# ------------------------------------------------------------
# Define prediction function
# This function accepts a driver's information
# as a Python dictionary and returns risk prediction
# ------------------------------------------------------------

def predict_driver_risk(driver_data):

    """
    Predict driver insurance risk using telematics data.

    Parameters:
        driver_data (dict):
            Dictionary containing driver information.

    Returns:
        dict:
            Predicted label, category, and confidence score.
    """

    # --------------------------------------------------------
    # Apply completeness override rule
    # If trip completeness is below 60 percent,
    # immediately classify as High Risk
    # This prevents selective disconnection fraud
    # --------------------------------------------------------

    if driver_data["trip_completeness_pct"] < 60:

        return {
            "predicted_label": 2,
            "predicted_category": "High Risk",
            "confidence_score": 1.0
        }

    # --------------------------------------------------------
    # Convert input dictionary into DataFrame
    # Models expect structured tabular input
    # --------------------------------------------------------

    input_df = pd.DataFrame([driver_data])

    # --------------------------------------------------------
    # Encode vehicle_type using saved LabelEncoder
    # Machine learning models require numeric input
    # --------------------------------------------------------

    input_df["vehicle_type"] = encoder.transform(
        input_df["vehicle_type"]
    )

    # --------------------------------------------------------
    # Define feature order
    # Feature order must match training data order
    # --------------------------------------------------------

    feature_columns = [

        "age",
        "driving_experience_yr",
        "past_claims",
        "traffic_violations",
        "vehicle_type",
        "vehicle_age_yr",
        "engine_cc",
        "avg_speed_kmh",
        "harsh_braking_per100km",
        "rapid_accel_per100km",
        "cornering_force_avg_g",
        "speeding_pct",
        "night_driving_pct",
        "total_trips",
        "avg_trip_km",
        "trip_completeness_pct",
        "unverified_trips",
        "days_since_policy_start",
        "onboarding_phase"
    ]

    # --------------------------------------------------------
    # Reorder DataFrame columns
    # This ensures consistency during prediction
    # --------------------------------------------------------

    input_df = input_df[feature_columns]

    # --------------------------------------------------------
    # Scale numerical feature values
    # The same scaler used during training is reused
    # --------------------------------------------------------

    scaled_input = scaler.transform(input_df)

    # --------------------------------------------------------
    # Predict risk label using trained model
    # --------------------------------------------------------

    predicted_label = model.predict(scaled_input)[0]

    # --------------------------------------------------------
    # Predict probability scores for all classes
    # predict_proba returns probability distribution
    # --------------------------------------------------------

    probabilities = model.predict_proba(scaled_input)[0]

    # --------------------------------------------------------
    # Extract highest confidence probability
    # --------------------------------------------------------

    confidence_score = round(
        np.max(probabilities),
        4
    )

    # --------------------------------------------------------
    # Convert numerical label into readable category
    # --------------------------------------------------------

    label_mapping = {
        0: "Low Risk",
        1: "Medium Risk",
        2: "High Risk"
    }

    predicted_category = label_mapping[predicted_label]

    # --------------------------------------------------------
    # Return final prediction results
    # --------------------------------------------------------

    return {

        "predicted_label": int(predicted_label),

        "predicted_category": predicted_category,

        "confidence_score": confidence_score
    }

# ------------------------------------------------------------
# Sample prediction test
# This example represents a realistic medium-risk
# Indian driver with moderate behavioural risk
# ------------------------------------------------------------

sample_driver = {

    "age": 34,

    "driving_experience_yr": 10,

    "past_claims": 1,

    "traffic_violations": 1,

    "vehicle_type": "Hatchback",

    "vehicle_age_yr": 6,

    "engine_cc": 1200,

    "avg_speed_kmh": 58.5,

    "harsh_braking_per100km": 8.2,

    "rapid_accel_per100km": 7.4,

    "cornering_force_avg_g": 0.24,

    "speeding_pct": 18.0,

    "night_driving_pct": 14.5,

    "total_trips": 65,

    "avg_trip_km": 18.3,

    "trip_completeness_pct": 92,

    "unverified_trips": 1,

    "days_since_policy_start": 180,

    "onboarding_phase": 1
}

# ------------------------------------------------------------
# Call prediction function using sample data
# ------------------------------------------------------------

prediction_result = predict_driver_risk(
    sample_driver
)

# ------------------------------------------------------------
# Print final prediction result
# ------------------------------------------------------------

print("\n================ DRIVER RISK PREDICTION ================\n")

print(prediction_result)