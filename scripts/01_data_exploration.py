# ============================================================
# FILE: 01_data_exploration.py
# PROJECT: Auto Insurance Risk Assessment using Telematics Data
# ============================================================

# ------------------------------------------------------------
# Import required libraries
# Pandas is used for data handling and analysis
# Matplotlib and Seaborn are used for data visualization
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Import Patch for creating a custom legend
# This ensures the legend colors correctly match
# the risk category colors in the chart
# ------------------------------------------------------------

from matplotlib.patches import Patch

# ------------------------------------------------------------
# Set Seaborn style for cleaner visual appearance
# ------------------------------------------------------------

sns.set_style("whitegrid")

# ------------------------------------------------------------
# Load the dataset using Pandas
# The CSV file contains telematics, driver, vehicle,
# connectivity, and risk classification information
# ------------------------------------------------------------

df = pd.read_csv("telematics_sample_dataset.csv")

# ------------------------------------------------------------
# Print the shape of the dataset
# Shape shows the number of rows and columns
# ------------------------------------------------------------

print("\n================ DATASET SHAPE ================\n")

print(df.shape)

# ------------------------------------------------------------
# Print all column names
# This helps understand available features
# ------------------------------------------------------------

print("\n================ COLUMN NAMES ================\n")

print(df.columns.tolist())

# ------------------------------------------------------------
# Print summary statistics using describe()
# This provides mean, standard deviation,
# minimum, and maximum values
# ------------------------------------------------------------

print("\n================ BASIC STATISTICS ================\n")

print(df.describe())

# ------------------------------------------------------------
# Check missing values in each column
# Missing values can affect ML model performance
# ------------------------------------------------------------

print("\n================ MISSING VALUES ================\n")

print(df.isnull().sum())

# ------------------------------------------------------------
# Print risk category distribution counts
# This helps identify class imbalance
# ------------------------------------------------------------

print("\n================ RISK CATEGORY COUNTS ================\n")

print(df["risk_category"].value_counts())

# ------------------------------------------------------------
# Create a bar chart for risk category distribution
# Fixed category order and fixed colors are used
# to avoid legend mismatch issues
# ------------------------------------------------------------

# Define correct risk category order
risk_order = [
    "Low Risk",
    "Medium Risk",
    "High Risk"
]

# Define fixed colors for each category
risk_colors = {
    "Low Risk": "green",
    "Medium Risk": "orange",
    "High Risk": "red"
}

# Create figure
plt.figure(figsize=(8, 5))

# Create count plot
sns.countplot(
    x="risk_category",
    data=df,
    order=risk_order,
    palette=risk_colors
)

# Add chart title and labels
plt.title(
    "Risk Category Distribution",
    fontsize=14
)

plt.xlabel(
    "Risk Category",
    fontsize=12
)

plt.ylabel(
    "Number of Drivers",
    fontsize=12
)

# ------------------------------------------------------------
# Create custom legend
# This ensures colors correctly match labels
# ------------------------------------------------------------

legend_elements = [

    Patch(
        facecolor="green",
        label="Low Risk"
    ),

    Patch(
        facecolor="orange",
        label="Medium Risk"
    ),

    Patch(
        facecolor="red",
        label="High Risk"
    )
]

plt.legend(
    handles=legend_elements,
    title="Risk Levels"
)

# Improve spacing
plt.tight_layout()

# Save chart
plt.savefig(
    "risk_category_distribution.png",
    dpi=300
)

# Close plot
plt.close()

# ------------------------------------------------------------
# Create correlation heatmap for numerical features
# Correlation helps identify relationships
# between numerical variables
# ------------------------------------------------------------

# Select only numerical columns
numerical_df = df.select_dtypes(
    include=["int64", "float64"]
)

# Calculate correlation matrix
correlation_matrix = numerical_df.corr()

# Create figure
plt.figure(figsize=(14, 10))

# Create heatmap
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f"
)

# Add title
plt.title(
    "Correlation Heatmap of Numerical Features",
    fontsize=14
)

# Improve spacing
plt.tight_layout()

# Save heatmap
plt.savefig(
    "correlation_heatmap.png",
    dpi=300
)

# Close plot
plt.close()

# ------------------------------------------------------------
# Create box plot for harsh braking grouped
# by risk category
# This helps compare braking behaviour
# across different risk levels
# ------------------------------------------------------------

# Create figure
plt.figure(figsize=(8, 5))

# Create box plot
sns.boxplot(
    x="risk_category",
    y="harsh_braking_per100km",
    data=df,
    order=risk_order,
    palette=risk_colors
)

# Add title and labels
plt.title(
    "Harsh Braking per 100km by Risk Category",
    fontsize=14
)

plt.xlabel(
    "Risk Category",
    fontsize=12
)

plt.ylabel(
    "Harsh Braking per 100km",
    fontsize=12
)

# Improve spacing
plt.tight_layout()

# Save box plot
plt.savefig(
    "braking_by_risk.png",
    dpi=300
)

# Close plot
plt.close()

# ------------------------------------------------------------
# Final completion message
# ------------------------------------------------------------

print("\nData exploration completed successfully.")

print("\nCharts saved successfully:")

print("1. risk_category_distribution.png")

print("2. correlation_heatmap.png")

print("3. braking_by_risk.png")