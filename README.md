# Telco Customer Churn Prediction Project

## 1. Project Overview

This is an end-to-end data science project focused on predicting customer churn for a telecommunications company. The primary goal is to build a machine learning model that can accurately identify customers who are at a high risk of leaving the service. By understanding the key drivers of churn, the company can implement targeted retention strategies to reduce revenue loss.

This project demonstrates a complete workflow, including data cleaning, exploratory data analysis (EDA), handling class imbalance with SMOTE, and building a high-performance classification model. The final Random Forest model serves as a powerful tool for proactive customer retention.

---

## 2. The Dataset

The data for this project was sourced from the popular "Telco Customer Churn" dataset available on Kaggle.

- **Source:** [IBM Business Analytics Telco Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Content:** The dataset contains 7,043 customer records with 21 features, including:
  - **Customer Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
  - **Account Information:** `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
  - **Subscribed Services:** `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`, etc.
  - **Target Variable:** `Churn` (Yes/No).

---

## 3. Methodology

The project was structured into a clear, multi-stage process:

1.  **Data Cleaning & Preprocessing:**
    - Corrected the `TotalCharges` column, converting it to a numeric type and handling missing values for new customers (tenure=0).
    - Dropped the non-informative `customerID` column.
    - Transformed all categorical text-based features into a numerical format using one-hot encoding (`pd.get_dummies`) to prepare them for the model.

2.  **Exploratory Data Analysis (EDA):**
    - Visual analysis revealed a significant class imbalance, with churned customers representing a minority.
    - The most critical insight discovered was the strong correlation between **Contract Type** and churn. Customers on `Month-to-month` contracts have a dramatically higher churn rate compared to those on `One year` or `Two year` contracts.
    - Other factors like low `tenure` and `Fiber optic` internet service were also identified as significant contributors to churn.

    *(Here are some key visualizations from the analysis):*

    ![Churn Rate by Contract Type](reports/figures/churn_by_contract.png) <!-- STEP 1: Save this plot image to this folder -->
    
3.  **Modeling & Evaluation:**
    - The dataset was split into training (80%) and testing (20%) sets.
    - To address the class imbalance, the **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data, creating a balanced set for the model to learn from.
    - A `RandomForestClassifier` was trained on the resampled data.
    - The model's performance was evaluated on the unseen test set, with a focus on its ability to correctly identify customers who churned (Recall).

---

## 4. Results & Performance

The final model, trained on the SMOTE-balanced data, demonstrated strong predictive power.

- **Confusion Matrix:** *(The model correctly identified 221 churners while missing 153, a significant improvement over the baseline model.)*

  ![SMOTE Model Confusion Matrix](reports/figures/smote_confusion_matrix.png) <!-- STEP 2: Save this plot image to this folder -->

- **Key Finding:** The model's ability to identify at-risk customers (Recall for 'Churned' class) was significantly improved by using SMOTE, making it a reliable tool for business action.

---

## 5. Tools & Libraries Used

- **Python 3.9**
- **Pandas & NumPy:** For data manipulation and numerical operations.
- **Matplotlib & Seaborn:** For data visualization and creating insightful plots.
- **Scikit-learn:** For core machine learning tasks (`train_test_split`, `RandomForestClassifier`).
- **Imbalanced-learn:** For handling class imbalance with `SMOTE`.
- **Jupyter Notebooks** within **Visual Studio Code** as the development environment.