import pandas as pd  # Import Pandas for data manipulation
import tensorflow as tf  # Import TensorFlow for building neural network models
import seaborn as sns  # Import Seaborn for statistical plotting
import matplotlib.pyplot as plt  # Import Matplotlib for creating plots
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting datasets
from sklearn.metrics import mean_squared_error, r2_score  # Import metrics for evaluating models
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNet  # Import linear regression models
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from sklearn.cross_decomposition import PLSRegression  # Import PLSRegression for partial least squares regression
from sklearn.preprocessing import SplineTransformer  # Import SplineTransformer for spline feature transformation
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # Import ensemble regression models
# Problem Set: Multi-Model Financial Prediction and Risk Analysis
# Student Exercise Manual
# Please complete the code

# -------------------------------
# Part 1: Data Generation and Preprocessing
# -------------------------------
from data_processing import split_data
X_train, X_val, X_test, y_train, y_val, y_test = split_data()

# -------------------------------
# Part 2: Model Training Section
# -------------------------------
from models.ols import train_ols_model
ols_model, y_pred_ols, r2_ols, mse_ols = train_ols_model(X_train, y_train, X_test, y_test)


# --- Weighted Linear Regression ---
   

# --- Huber Regressor ---


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
