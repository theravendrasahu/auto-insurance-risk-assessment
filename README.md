# 🚗 Auto Insurance Risk Assessment Using Machine Learning
### Powered by 3rd Generation Mobile Telematics Sensor Data | Built for Indian Drivers

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.9-red?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

**MSc Data Science Dissertation Project**

*Replacing demographic guessing with real driving behaviour*

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [The Problem We Are Solving](#-the-problem-we-are-solving)
- [How the System Works](#-how-the-system-works)
- [Key Results](#-key-results)
- [Dataset Description](#-dataset-description)
- [Feature Groups](#-feature-groups)
- [ML Models Compared](#-ml-models-compared)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Pipeline Flow](#-pipeline-flow)
- [System Design Highlights](#-system-design-highlights)
- [Sample Prediction](#-sample-prediction)
- [Output Files](#-output-files)
- [Technologies Used](#-technologies-used)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 🎯 Project Overview

This project is the practical implementation component of an MSc Data Science dissertation titled:

> **"Automated Risk Assessment for Auto Insurance Using Machine Learning Focused on 3rd Generation Telematics Sensor Data"**

Traditional auto insurance in India prices premiums based on **who you are** — your age, gender, and location. This project replaces that approach with a system that prices based on **how you actually drive**, using real sensor data collected from a smartphone.

The system classifies every driver into one of three risk categories:

| 🟢 Low Risk | 🟡 Medium Risk | 🔴 High Risk |
|---|---|---|
| Risk score below 0.35 | Risk score 0.35 to 0.62 | Risk score above 0.62 |
| Eligible for premium discount | Standard premium | Premium uplift applied |

---

## ❓ The Problem We Are Solving

### What is wrong with the current system?

```
Current System                          This Project
─────────────────────────────────────   ─────────────────────────────────────
Uses: Age, Gender, Location             Uses: Real driving behaviour data
Result: Group-based pricing             Result: Individual-based pricing
Fairness: Low                           Fairness: High
Fraud resistance: None                  Fraud resistance: Built-in
Accuracy: ~60-70% estimated             Accuracy: 94.2% measured
Transparency: Low                       Transparency: Full
```

### Three real problems this project addresses:

**Problem 1 — Unfair Pricing**
A careful 22-year-old driver pays the same premium as a reckless 22-year-old because the system only sees their age, not how they ride.

**Problem 2 — Cold Start Problem**
In India, insurance must be purchased on the day you buy a vehicle. But a telematics model needs driving data first. On day zero, there is no data. This project solves this with a phased onboarding approach.

**Problem 3 — Selective Connectivity Fraud**
A driver could deliberately disconnect their phone before harsh rides and reconnect during calm ones. This project detects and penalises this behaviour through a completeness override rule.

---

## ⚙️ How the System Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📱 Mobile Phone Sensors                                        │
│  ├── Accelerometer  → Detects braking and acceleration          │
│  ├── Gyroscope      → Detects cornering force                   │
│  └── GPS            → Tracks speed, location, timestamps        │
│           │                                                     │
│           ▼                                                     │
│  🔵 Bluetooth Verification                                      │
│  └── Confirms phone is INSIDE the vehicle before logging        │
│           │                                                     │
│           ▼                                                     │
│  🔄 Data Pipeline (6 Python Scripts)                           │
│  ├── 01 Data Exploration                                        │
│  ├── 02 Preprocessing + Fraud Rule                             │
│  ├── 03 Feature Engineering                                     │
│  ├── 04 Model Training (4 models)                              │
│  ├── 05 Model Evaluation + ROC                                 │
│  └── 06 Prediction Function                                    │
│           │                                                     │
│           ▼                                                     │
│  🤖 XGBoost Classifier (Best Model)                            │
│           │                                                     │
│           ▼                                                     │
│  📊 Risk Output: Low / Medium / High + Confidence Score        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Completeness Override Rule

```python
# If driver disconnects phone too often → automatic High Risk
if trip_completeness_pct < 60:
    return High Risk (confidence = 1.0)
# No model call needed — fraud is caught before prediction
```

---

## 📊 Key Results

### Model Performance on 1000-Record Test Set

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---|---|---|---|---|
| 🏆 **XGBoost** | **94.2%** | **0.94** | **0.94** | **0.94** | **0.995** |
| 🥈 Random Forest | 91.3% | 0.92 | 0.91 | 0.90 | 0.991 |
| 🥉 Decision Tree | 88.7% | 0.89 | 0.89 | 0.89 | 0.975 |
| Logistic Regression | 84.5% | 0.85 | 0.85 | 0.85 | 0.956 |

### XGBoost Confusion Matrix Summary

```
                  Predicted
                Low   Medium   High
Actual  Low  [  37      27       0  ]
       Med  [   2     514       6  ]
      High  [   0      23     391  ]

✅ Zero High Risk drivers classified as Low Risk
✅ 94.2% overall accuracy on unseen test data
✅ AUC of 0.995 — near perfect class separation
```

### Top 5 Most Important Features

```
1. trip_completeness_pct    ████████████████████████  0.52
2. harsh_braking_per100km   ███░░░░░░░░░░░░░░░░░░░░░  0.08
3. rapid_accel_per100km     ██░░░░░░░░░░░░░░░░░░░░░░  0.07
4. cornering_force_avg_g    █░░░░░░░░░░░░░░░░░░░░░░░  0.045
5. speeding_pct             █░░░░░░░░░░░░░░░░░░░░░░░  0.045
```

---

## 📁 Dataset Description

| Property | Value |
|---|---|
| Type | Synthetic — designed to reflect Indian driving conditions |
| Total Records | 5,000 drivers |
| Total Columns | 24 |
| Missing Values | 0 |
| Train Set | 4,000 records (80%) |
| Test Set | 1,000 records (20%) |
| Random Seed | 42 (fully reproducible) |
| Risk Distribution | Low: 320 — Medium: 2,611 — High: 2,069 |

### Why Synthetic Data?

Real telematics datasets for Indian drivers are not publicly available. Real insurance data is commercially sensitive. A synthetic dataset allowed full control over feature design to reflect Indian road conditions including dense city traffic, unpredictable road surfaces, and frequent harsh braking scenarios.

---

## 🧩 Feature Groups

### Group 1 — Driver Profile

| Column | Description | Range |
|---|---|---|
| age | Driver age in years | 18 to 61 |
| driving_experience_yr | Years of active driving | 0 to 40 |
| past_claims | Number of previous insurance claims | 0 to 3 |
| traffic_violations | Number of recorded violations | 0 to 2 |

### Group 2 — Vehicle Features

| Column | Description | Range |
|---|---|---|
| vehicle_type | Hatchback / Sedan / SUV / Two-Wheeler / Van | Categorical |
| vehicle_age_yr | Age of vehicle in years | 0 to 15 |
| engine_cc | Engine displacement in cubic centimetres | 100 to 2000 |

### Group 3 — Telematics Behaviour (from Mobile Sensors)

| Column | Sensor Source | Description |
|---|---|---|
| avg_speed_kmh | GPS | Average speed across all trips |
| harsh_braking_per100km | Accelerometer | Sudden braking events per 100km |
| rapid_accel_per100km | Accelerometer | Sharp acceleration events per 100km |
| cornering_force_avg_g | Gyroscope | Average lateral g-force during turns |
| speeding_pct | GPS | % of trip time over safe speed limit |
| night_driving_pct | GPS Timestamp | % of trips during night hours |
| total_trips | Aggregated | Total number of trips recorded |
| avg_trip_km | GPS | Average trip distance in km |

### Group 4 — Connectivity Features (Fraud Detection)

| Column | Description | Key Rule |
|---|---|---|
| trip_completeness_pct | % of OBD trips with mobile data connected | Below 60 → Auto High Risk |
| unverified_trips | Trips logged by OBD but missing mobile data | Higher = more suspicious |

### Group 5 — Policy Features

| Column | Description |
|---|---|
| days_since_policy_start | How many days since the policy began |
| onboarding_phase | 0 = Phase 1 flat premium, 1 = Phase 2 telematics active |

### Output Columns

| Column | Description |
|---|---|
| risk_score | Weighted formula score from 0 to 1 |
| risk_category | Low Risk / Medium Risk / High Risk |
| risk_label | 0 = Low / 1 = Medium / 2 = High |

---

## 🤖 ML Models Compared

### Logistic Regression — Baseline
- Finds a linear boundary between risk classes
- Fast, interpretable, serves as minimum benchmark
- Accuracy: 84.5% — AUC: 0.956

### Decision Tree — Interpretable Splits
- Builds a flowchart of if-then rules based on feature thresholds
- Max depth of 6 to prevent overfitting
- Can be drawn and explained to non-technical stakeholders
- Accuracy: 88.7% — AUC: 0.975

### Random Forest — Ensemble Stability
- Trains 100 decision trees on random subsets of data
- Final prediction by majority vote across all trees
- Much more stable than a single tree
- Accuracy: 91.3% — AUC: 0.991

### XGBoost — Best Model ✅
- Builds trees sequentially — each tree corrects errors of the previous
- Gradient boosting on the loss function at every step
- 100 estimators, learning rate 0.1, mlogloss evaluation
- Accuracy: 94.2% — AUC: 0.995

---

## 📂 Project Structure

```
auto-insurance-risk-assessment/
│
├── 📂 data/
│   └── telematics_sample_dataset.csv     # 5000-row synthetic dataset
│
├── 📂 scripts/
│   ├── 01_data_exploration.py            # EDA and visualisation
│   ├── 02_preprocessing.py              # Cleaning, encoding, scaling, splitting
│   ├── 03_feature_engineering.py        # Correlation check, feature importance
│   ├── 04_model_training.py             # Train 4 models, save best
│   ├── 05_model_evaluation.py           # ROC curves, metrics comparison
│   └── 06_predict.py                    # Prediction function for new drivers
│
├── 📂 outputs/
│   ├── risk_category_distribution.png   # Risk class bar chart
│   ├── correlation_heatmap.png          # Feature correlation heatmap
│   ├── braking_by_risk.png              # Box plot by risk category
│   ├── feature_importance.png           # Random Forest feature importance
│   ├── confusion_matrix_*.png           # One per model (4 files)
│   ├── roc_curves_comparison.png        # All 4 ROC curves on one chart
│   ├── model_metrics_comparison.png     # Grouped bar chart all metrics
│   └── dropped_features.txt            # Feature selection log
│
├── 📂 models/
│   └── best_model.pkl                   # Saved XGBoost model
│
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Files excluded from Git
└── README.md                            # This file
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.10 or above
- pip package manager

### Installation

```bash
# Step 1 — Clone the repository
git clone https://github.com/YourUsername/auto-insurance-risk-assessment.git

# Step 2 — Navigate into the project folder
cd auto-insurance-risk-assessment

# Step 3 — Install all required libraries
pip install -r requirements.txt
```

### Running the Pipeline

Run each script in order from the project root folder. Each script saves outputs that the next script depends on.

```bash
# Stage 1 — Explore the dataset and generate charts
python scripts/01_data_exploration.py

# Stage 2 — Clean, encode, scale and split the data
python scripts/02_preprocessing.py

# Stage 3 — Check correlations and extract feature importance
python scripts/03_feature_engineering.py

# Stage 4 — Train all 4 models and save the best one
python scripts/04_model_training.py

# Stage 5 — Generate ROC curves and comparison charts
python scripts/05_model_evaluation.py

# Stage 6 — Run a sample prediction
python scripts/06_predict.py
```

> ⚠️ **Important:** Run scripts from the project root folder, not from inside the scripts folder. The dataset path is relative to the root.

### Expected Console Output from Script 06

```
================ DRIVER RISK PREDICTION ================

{
  'predicted_label'    : 1,
  'predicted_category' : 'Medium Risk',
  'confidence_score'   : 0.87
}
```

---

## 🔄 Pipeline Flow

```
telematics_sample_dataset.csv
         │
         ▼
01_data_exploration.py
├── Prints shape, stats, missing values
├── Saves: risk_category_distribution.png
├── Saves: correlation_heatmap.png
└── Saves: braking_by_risk.png
         │
         ▼
02_preprocessing.py
├── Drops non-feature columns
├── Handles missing values
├── Applies completeness override rule
├── Encodes vehicle_type
├── Scales all numerical features
├── Splits 80% train / 20% test
└── Saves: X_train.csv, X_test.csv,
          y_train.csv, y_test.csv,
          scaler.pkl, encoder.pkl
         │
         ▼
03_feature_engineering.py
├── Checks correlations (threshold 0.85)
├── Trains Random Forest for importance
├── Drops redundant correlated features
└── Saves: feature_importance.png,
          dropped_features.txt,
          X_train_feature_selected.csv
         │
         ▼
04_model_training.py
├── Trains: Logistic Regression
├── Trains: Decision Tree (max_depth=6)
├── Trains: Random Forest (100 trees)
├── Trains: XGBoost (100 est, lr=0.1)
├── Evaluates all on test set
├── Saves: 4 confusion matrix PNGs
└── Saves: best_model.pkl (XGBoost)
         │
         ▼
05_model_evaluation.py
├── Reloads all models
├── Generates ROC curves (AUC scores)
├── Generates metrics comparison chart
└── Saves: roc_curves_comparison.png,
          model_metrics_comparison.png
         │
         ▼
06_predict.py
├── Loads: scaler.pkl, encoder.pkl,
│         best_model.pkl
├── Applies completeness override rule
├── Encodes and scales new input
└── Returns: risk label + confidence score
```

---

## 🛡️ System Design Highlights

### Phased Onboarding — Solving the Cold Start Problem

```
Day 0 (Vehicle Purchase)
└── Standard flat demographic premium charged
└── Sensor data collection begins silently

Days 1 to 90 — Phase 1
└── Policy fully valid, all claims covered
└── No telematics scoring yet, just data collection

Day 91 onwards — Phase 2
└── Model activated with real driving data
└── Telematics-based premium calculated
└── Premium adjusted based on actual behaviour
```

### Completeness Override — Fraud Prevention

```
Every Trip
└── OBD dongle logs trip independently
└── Mobile sensors log trip independently
└── System cross-references both streams

End of Month Check:
trip_completeness_pct = (verified trips / total OBD trips) × 100

If trip_completeness_pct < 60%:
└── → Automatic High Risk regardless of riding quality
└── → No model call needed
└── → Cannot be gamed by selective disconnection
```

---

## 🔮 Sample Prediction

```python
from scripts.predict import predict_driver_risk

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

result = predict_driver_risk(sample_driver)
print(result)
# Output:
# {
#   'predicted_label': 1,
#   'predicted_category': 'Medium Risk',
#   'confidence_score': 0.87
# }
```

---

## 📤 Output Files

| File | Description |
|---|---|
| `risk_category_distribution.png` | Bar chart showing count of each risk class |
| `correlation_heatmap.png` | Heatmap of correlations between all 20 numerical features |
| `braking_by_risk.png` | Box plot comparing braking rates across risk categories |
| `feature_importance.png` | Horizontal bar chart of all 19 feature importance scores |
| `confusion_matrix_Logistic_Regression.png` | Confusion matrix for LR model |
| `confusion_matrix_Decision_Tree.png` | Confusion matrix for DT model |
| `confusion_matrix_Random_Forest.png` | Confusion matrix for RF model |
| `confusion_matrix_XGBoost.png` | Confusion matrix for XGBoost model |
| `roc_curves_comparison.png` | All 4 ROC curves with AUC scores on one chart |
| `model_metrics_comparison.png` | Grouped bar chart comparing all metrics |
| `dropped_features.txt` | Log confirming no features were dropped |
| `best_model.pkl` | Saved XGBoost model ready for prediction |

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10 | Core programming language |
| Pandas | 2.0 | Data loading and manipulation |
| NumPy | Latest | Numerical operations |
| Scikit-learn | 1.3 | ML models, preprocessing, metrics |
| XGBoost | 1.9 | Gradient boosting classifier |
| Matplotlib | Latest | Chart generation and saving |
| Seaborn | Latest | Statistical visualisations |
| Joblib | Latest | Model and object persistence |

---

## 🔭 Future Enhancements

| Enhancement | Description |
|---|---|
| 📱 In-Vehicle Android App | Install directly on built-in Android infotainment systems in modern cars — no phone needed |
| 🔴 Real-Time IoT Streaming | Stream sensor data live during trips using MQTT protocol |
| 📲 Mobile App Dashboard | Native app with live risk score, trip completeness, and premium preview |
| 📅 Monthly Rolling Premium | Recalculate premium every 30 days based on recent behaviour only |
| 🧠 LSTM Deep Learning | Sequential model to learn patterns from second-by-second trip data |
| ☁️ Cloud Deployment | REST API on AWS or GCP for national-scale operation |
| 🔍 Anomaly Detection | Detect behavioural fingerprint changes to identify sophisticated gaming |
| 🌦️ Weather Context | Adjust risk scores based on real-time road and weather conditions |
| 🚨 Crash Detection | Auto-detect accidents from sensor spikes and file insurance claims |
| 🔗 Insurance API | Connect directly to insurer policy systems for automatic premium updates |

---

## 👤 Author

**Ravendra Sahu**
MSc Data Science Student
Cotality — Data Analyst

📧 Connect on LinkedIn: [https://www.linkedin.com/in/theravendrasahu/]
🐙 GitHub: [https://github.com/theravendrasahu]

---

## 📄 License

This project is submitted as part of an MSc Data Science dissertation.
The code is open for academic reference and personal learning.

---

<div align="center">

**If this project helped you, please give it a ⭐ on GitHub**

*Built with Python 🐍 | Powered by XGBoost 🚀 | Made for Indian Roads 🇮🇳*

</div>
