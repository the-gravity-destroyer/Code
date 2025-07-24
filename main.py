# Problem Set: Multi-Model Financial Prediction and Risk Analysis
# Student Exercise Manual
# Please complete the code

# -------------------------------
# Part 1 & 2: Data Generation and Preprocessing
# -------------------------------
from data_processing import split_data
X_train, X_val, X_test, y_train, y_val, y_test = split_data()

# -------------------------------
# Part 3: Model Training Section
# -------------------------------

# --- OLS Model Training and Evaluation ---
from models.ordinary_least_squares_regression import train_ols_model
ols_model, y_pred_ols, r2_ols, mse_ols = train_ols_model(X_train, y_train, X_val, y_val, X_test, y_test)

# --- ElasticNet Model Tuning ---
from models.elastic_net_regression import train_elastic_net_model
elastic_net_model, y_pred_en, r2_en, mse_en = train_elastic_net_model(X_train, y_train, X_test, y_test)

# --- Principal Component Regression (PCR) ---
from models.principal_component_regression import train_pcr_model
pcr = train_pcr_model()

# --- Partial Least Squares Regression (PLS) ---
from models.partial_least_squares_regression import train_pls_model
pls_model, y_pred_pls, r2_pls, mse_pls = train_pls_model(X_train, y_train, X_test, y_test)

# --- Generalized Linear Model (Spline Transformation + ElasticNet) ---
from models.generalized_linear_model import train_glm_model
glm_model, y_pred_glm, r2_glm, mse_glm = train_glm_model(X_train, y_train, X_test, y_test)

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
