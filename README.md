# 🍷 Wine Analysis & Quality Prediction API

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

A machine learning project designed to analyze wine chemical properties, classify wine varieties, and provide quality assessments via a REST API.
**Wine Analysis** allows users to evaluate wine characteristics (specifically pH levels) against standard oenological rules to ensure quality and consistency.

## ✨ Features

- **Data Analysis:** Comprehensive EDA pipeline including outlier detection (IQR) and feature correlation analysis.
- **Machine Learning:** Comparative training of multiple models (Logistic Regression, SVM, KNN) with **Random Forest** achieving the best results.
- **Data Preprocessing:** Implements SMOTE for class balancing and StandardScaler for normalization.
- **FastAPI Backend:** A lightweight, fast API to serve predictions and evaluate wine quality rules in real-time.
- **Rule-Based Logic:** Automatic verification of pH suitability for Red and White wines.

## 📂 Project Structure

```text
Wine-Project/
├── Wine Type predicting.ipynb   # Data Science Pipeline (EDA, Training, Evaluation)
├── main.py                      # FastAPI Server Entry Point
├── Wine_Dataset.csv             # Source Dataset
├── wine_quality_model.pkl       # Trained Model Artifact
├── best_model.pkl               # Best performing model from GridSearch
└── README.md
