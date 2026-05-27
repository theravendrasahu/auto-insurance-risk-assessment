# Auto Insurance Risk Assessment Using Machine Learning

## MSc Data Science Dissertation Project
**Author:** Ravendra Sahu  
**Institution:** [Your University Name]  
**Year:** 2025

---

## Project Overview

This project builds a machine learning pipeline to assess 
auto insurance risk for Indian drivers using third-generation 
mobile telematics sensor data. Drivers are classified into 
Low Risk, Medium Risk, or High Risk categories based on their 
actual driving behaviour collected from smartphone sensors 
including accelerometer, gyroscope, and GPS.

The system uses Bluetooth vehicle verification to confirm 
that sensor data is captured from inside the vehicle, and 
includes a completeness override rule to prevent selective 
disconnection fraud.

---

## Models Compared

| Model | Accuracy | AUC |
|---|---|---|
| XGBoost (Best) | 94.2% | 0.995 |
| Random Forest | 91.3% | 0.991 |
| Decision Tree | 88.7% | 0.975 |
| Logistic Regression | 84.5% | 0.956 |

---

## Dataset

- **Type:** Synthetic dataset reflecting Indian driving conditions
- **Rows:** 5000 drivers
- **Columns:** 24 features
- **Features:** Driver profile, vehicle characteristics, 
  telematics behaviour, connectivity quality, policy phase

---

## Project Structure
