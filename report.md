
# AI Expense Analysis - Report

Goal: Identify patterns in monthly expenses and predict next-month spending per category using a Decision Tree regressor.

Data: Synthetic monthly expense data for 12 months across categories: Food, Transport, Entertainment, Utilities, Healthcare.

Approach:
- Aggregate raw transactions to monthly sums per category.
- Create supervised problem using previous 2 months as features to predict the next month's spending vector (multi-output).
- Train a DecisionTreeRegressor (multi-output) and evaluate with R² and MAE.
- Predict next month's spending and visualize history + prediction for top categories.

Files included:
- data.csv - sample transactions
- ai_expense_analysis.py - analysis and model
- report.md - this document

How to run:
1. Make sure you have Python 3.8+ and install dependencies:
   pip install pandas scikit-learn matplotlib
2. Run:
   python ai_expense_analysis.py

Notes & next steps:
- Add more features: marketing spend, income, holidays/seasonality flags.
- Try ensemble models (RandomForest, GradientBoosting) and cross-validation.
- Add hyperparameter tuning and feature importance analysis.
