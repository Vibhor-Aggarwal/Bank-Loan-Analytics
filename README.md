# Loan Portfolio Status Prediction & Analytics

## Project Overview

Loan Portfolio Status Prediction & Analytics is an end-to-end Machine Learning and Data Analytics project that predicts whether a loan is likely to be **Fully Paid** or **Charged Off** using historical Lending Club loan data.

The project follows an industry-style data science workflow, including data understanding, data cleaning, exploratory data analysis (EDA), preprocessing, feature engineering, model comparison, hyperparameter tuning, threshold optimization, and feature importance analysis.

Beyond predictive modeling, the project is designed to demonstrate the complete analytics lifecycle by incorporating SQL-based business analysis, interactive Power BI dashboards, and a Streamlit web application.

The objective is not only to build an accurate prediction model but also to generate actionable business insights that can support better lending decisions and portfolio risk management.

---

# Business Problem

Financial institutions process thousands of loan applications every day. Approving high-risk borrowers can lead to significant financial losses, while rejecting reliable borrowers can reduce business opportunities.

This project aims to:

* Predict whether a loan will be Fully Paid or Charged Off.
* Identify the key factors influencing loan repayment.
* Analyze customer and loan characteristics.
* Support data-driven lending decisions.
* Demonstrate an end-to-end machine learning workflow suitable for industry applications.

---

# Dataset Information

**Dataset:** Lending Club Loan Dataset

**Target Variable**

* `loan_status`

Classes:

* Fully Paid (Healthy Loan)
* Charged Off (Risky Loan)

Dataset Size:

* **396,030 records**

Training Set:

* **316,824 rows**

Testing Set:

* **79,206 rows**

Processed Features:

* **48 features**

---

# Project Workflow

```text
Raw Dataset
      │
      ▼
Data Understanding
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Preprocessing
      │
      ▼
Train-Test Split
      │
      ▼
Feature Encoding
      │
      ▼
Model Comparison
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Threshold Optimization
      │
      ▼
Feature Importance Analysis
      │
      ▼
Final Model
```

---

# Project Structure

```text
Loan-Portfolio-Status-Prediction-Analytics/

│
├── data/
│   ├── raw/
│   └── processed/
│
├── images/
│
├── models/
│
├── reports/
│
├── src/
│
├── README.md
│
└── requirements.txt
```

---

# Tech Stack

## Programming

* Python

## Data Analysis

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* Imbalanced-learn (SMOTE)
* Random Forest Classifier
* RandomizedSearchCV

## Version Control

* Git
* GitHub

## Business Intelligence *(Planned)*

* SQL
* Power BI

## Deployment *(Planned)*

* Streamlit

---

# Exploratory Data Analysis (EDA)

The dataset was explored to better understand borrower characteristics and loan behavior.

EDA included:

* Missing value analysis
* Duplicate detection
* Distribution analysis
* Correlation analysis
* Loan status visualization
* Loan grade analysis
* Interest rate analysis
* Income distribution
* Home ownership analysis
* Loan purpose analysis
* Numerical feature distributions
* Categorical feature analysis

Several visualizations were generated and saved in the **images/** directory.

---

# Data Preprocessing

The preprocessing pipeline included:

* Handling missing values
* Removing unnecessary columns
* Encoding categorical variables
* Feature scaling where required
* Train-test splitting
* Saving processed datasets

Generated files:

```text
data/processed/

x_train.csv
x_test.csv
y_train.csv
y_test.csv
```

---

# Machine Learning Models

Multiple classification models were trained and compared.

The project evaluates different algorithms before selecting the best-performing model.

Performance comparison is saved in:

```text
reports/model_comparison.csv
```

---

# Hyperparameter Tuning

The final Random Forest model was optimized using:

* RandomizedSearchCV
* Cross Validation
* SMOTE
* F1 Score optimization

Search configuration:

* n_iter = 5
* cv = 3

The best model is saved as:

```text
models/best_random_forest.pkl
```

Hyperparameter results are saved in:

```text
reports/hyperparameter_results.csv
```

---

# Threshold Optimization

Instead of relying only on the default prediction threshold (0.50), the model evaluates multiple probability thresholds:

* 0.50
* 0.40
* 0.30
* 0.20
* 0.10

For each threshold, the following metrics are calculated:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The threshold with the highest F1 Score is selected for the final evaluation.

---

# Feature Importance Analysis

The trained Random Forest model was analyzed to identify the most influential features affecting loan predictions.

Top contributing features include:

* Grade
* Interest Rate
* Sub Grade
* Debt Consolidation Purpose
* Loan Term
* Home Ownership
* Verification Status
* Debt-to-Income Ratio

Outputs generated:

```text
reports/feature_importance.csv

images/feature_importance.png
```

These insights improve model interpretability and help explain prediction behavior to business stakeholders.

---

# Results

The project successfully demonstrates:

* End-to-end machine learning workflow
* Data cleaning and preprocessing
* Exploratory data analysis
* Model comparison
* Hyperparameter tuning
* Threshold optimization
* Feature importance analysis
* Professional project organization
* Business-oriented interpretation of results

---

# Future Improvements

The following enhancements are planned:

* SQL business analysis
* Power BI dashboard
* Streamlit web application
* Model deployment
* SHAP-based explainability
* Additional machine learning models
* Automated prediction interface

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<Vibhor-Aggarwal>>/Loan-Portfolio-Status-Prediction-Analytics.git
```

Navigate to the project directory:

```bash
cd Loan-Portfolio-Status-Prediction-Analytics
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows**

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run individual modules from the project root:

```bash
python src/data_understanding.py
```

```bash
python src/data_cleaning.py
```

```bash
python src/eda.py
```

```bash
python src/preprocessing.py
```

```bash
python src/model_comparison.py
```

```bash
python src/hyperparametertuning.py
```

```bash
python src/feature_importance.py
```

---

# Repository Highlights

* Professional folder structure
* Modular Python scripts
* Well-documented functions
* Reproducible workflow
* Business-focused machine learning pipeline
* Git version control
* Interview-ready portfolio project

---

# Author

**Vibhor Aggarwal**

Computer Science Engineering (FinTech)

Python | Data Analytics | Machine Learning

GitHub: https://github.com/<your-username>

---

# License

This project is intended for educational, learning, and portfolio purposes.
