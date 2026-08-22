# Customer Churn Prediction Using Machine Learning and XGBoost

An end-to-end machine learning project for predicting telecom customer churn using exploratory data analysis, preprocessing, Decision Trees, Random Forest, XGBoost, cross-validation, and hyperparameter tuning.

## Project Overview

Customer churn occurs when customers stop using a company's services.

The goal of this project is to predict which telecom customers are likely to churn and identify the customer characteristics most strongly associated with churn.

The project uses the IBM Telco Customer Churn dataset and compares several machine learning models before selecting and tuning XGBoost.

## Dataset

The dataset contains 7,043 customer observations and includes information about:

- Customer demographics
- Contract type
- Internet and phone services
- Technical support and online security
- Payment method
- Monthly and total charges
- Customer tenure
- Churn status

## Project Workflow

1. Data loading and inspection
2. Data cleaning
3. Exploratory Data Analysis
4. Target and feature preparation
5. Train-test split
6. One-hot encoding
7. Baseline model
8. Decision Tree
9. Random Forest
10. XGBoost
11. 5-fold cross-validation
12. Hyperparameter tuning with RandomizedSearchCV
13. Final test-set evaluation
14. Confusion matrix analysis
15. Feature importance
16. Business insights

## Models Compared

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Decision Tree | 72.15% | 47.63% | 48.83% | 48.19% |
| Random Forest | 78.70% | 63.07% | 47.76% | 54.33% |
| XGBoost | 78.42% | 60.96% | 51.97% | 56.09% |

## Tuned XGBoost Results

The final tuned XGBoost model achieved:

- **Accuracy:** 79.91%
- **Precision:** 65.85%
- **Recall:** 50.53%
- **F1-score:** 57.19%

The tuned model correctly identified approximately half of the customers who actually churned while maintaining relatively strong precision.

## Best Hyperparameters

```python
{
    "subsample": 0.8,
    "n_estimators": 300,
    "max_depth": 2,
    "learning_rate": 0.05,
    "colsample_bytree": 0.7
}
