# Problem Set: Multi-Model Financial Prediction and Risk Analysis
# Student Exercise Manual
# Please complete the code

from models.linear_models import ordinary_least_squares_regression as ols


# -------------------------------
# Part 1 & 2: Data Generation and Preprocessing
# -------------------------------
from data_processing import split_data
X_train, X_val, X_test, y_train, y_val, y_test = split_data()

# -------------------------------
# Part 3: Model Training Section
# -------------------------------

# --- OLS Model Training and Evaluation ---
ols_model = ols.OLSModel(n_stocks=10)
ols_model.train(X_train, y_train)
val_metrics = ols_model.evaluate(X_val, y_val)
ols_model.print_summary("OLS Validation", val_metrics)
test_metrics = ols_model.evaluate(X_test, y_test)
ols_model.print_summary("OLS Test", test_metrics)
ols_model.print_feature_importance(top_n=10)
ols_model.plot_diagnostics(X_test, y_test)

# --- ElasticNet Model Tuning ---

# --- Principal Component Regression (PCR) ---

# --- Partial Least Squares Regression (PLS) ---

# --- Generalized Linear Model (Spline Transformation + ElasticNet) ---

# --- non-linear models ---

# --- Neural Network Model ---

# --- Gradient Boosting Regressor ---

# --- Random Forest Regressor ---


# -------------------------------
# Part 4: Prediction Wrappers
# -------------------------------

# -------------------------------
# Part 5: Full-Sample Time Series Plots - to see the predictions vs. actuals
# -------------------------------

# -------------------------------
# Part 6: Out-of-Sample R² Results Table - to evaluate model performance
# -------------------------------
# TODO: Calculate R² according to the formula: 1 - (sum of squared errors / total sum of squares)

# -------------------------------
# Part 7: Diebold-Mariano Test Statistics - to compare model predictions
# -------------------------------
# -------------------------------
# Part 8: Variable Importance Calculations & Heatmaps - to understand feature importance ( to see which features are more important)
# -------------------------------
# TODO: Define a function to compute variable importance based on the drop in R² when a feature is removed
# -------------------------------
# -------------------------------
# Part 9: Auxiliary Functions and Decile Portfolio Analysis - to analyze model performance across deciles - to compare predicted vs actual  sharpe ratios
# -------------------------------
