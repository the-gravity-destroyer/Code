# Problem Set: Multi-Model Financial Prediction and Risk Analysis
# Student Exercise Manual
# Please complete the code

from models.linear_models.ordinary_least_squares_regression import OLSModel
from models.linear_models.elastic_net_regression import ElasticNetModel
from models.linear_models.principal_component_regression import PCRModel
from models.linear_models.partial_least_squares_regression import PLSModel
from models.linear_models.generalized_linear_model import GLMModel
from models.non_linear_models.neural_network import NeuralNetworkModel
#from models.non_linear_models.gradient_boosting import GradientBoostingModel
#from models.non_linear_models.random_forest import RandomForestModel

from data_processing import split_data


# -------------------------------
# Part 1 & 2: Data Generation and Preprocessing
# -------------------------------
X_train, X_val, X_test, y_train, y_val, y_test = split_data()

# -------------------------------
# Part 3: Model Training Section
# -------------------------------

# --- OLS Model Training and Evaluation ---
'''
ols_model = OLSModel(n_stocks=10)
ols_model.train(X_train, y_train)

# --- ElasticNet Model Tuning ---
elastic_net_model = ElasticNetModel(n_stocks=10)
elastic_net_model.train(X_train, y_train, X_val, y_val)

# --- Principal Component Regression (PCR) ---
pcr_model = PCRModel(n_stocks=10)
pcr_model.train(X_train, y_train, X_val, y_val)

# --- Partial Least Squares Regression (PLS) ---
pls_model = PLSModel(n_stocks=10)
pls_model.train(X_train, y_train, X_val, y_val)

# --- Generalized Linear Model (Spline Transformation + ElasticNet) ---
glm = GLMModel(n_stocks=10)
glm.train(X_train, y_train, X_val, y_val)

# --- non-linear models ---

# --- Neural Network Model ---
nn_model = NeuralNetworkModel(n_stocks=10)
nn_model.train(X_train, y_train, X_val, y_val)
'''


# --- Gradient Boosting Regressor ---

# --- Random Forest Regressor ---


# -------------------------------
# Part 4: Prediction Wrappers
# -------------------------------

# -------------------------------
# Part 5: Full-Sample Time Series Plots - to see the predictions vs. actuals
# -------------------------------
'''
ols_model.plot_diagnostics(X_test, y_test)
elastic_net_model.plot_diagnostics(X_test, y_test)
pcr_model.plot_diagnostics(X_test, y_test)
pls_model.plot_diagnostics(X_test, y_test)
glm.plot_diagnostics(X_test, y_test)
nn_model.plot_diagnostics(X_test, y_test)
'''


# -------------------------------
# Part 6: Out-of-Sample R² Results Table - to evaluate model performance
# -------------------------------
# TODO: Calculate R² according to the formula: 1 - (sum of squared errors / total sum of squares)
'''
val_metrics = ols_model.evaluate(X_val, y_val)
ols_model.print_summary("OLS Validation", val_metrics)<
test_metrics = ols_model.evaluate(X_test, y_test)
ols_model.print_summary("OLS Test", test_metrics)
ols_model.print_feature_importance(top_n=10)

elastic_net_model.print_hyperparameters()
val_metrics = elastic_net_model.evaluate(X_val, y_val)
elastic_net_model.print_summary("ElasticNet Validation", val_metrics)
test_metrics = elastic_net_model.evaluate(X_test, y_test)
elastic_net_model.print_summary("ElasticNet Test", test_metrics)
elastic_net_model.print_feature_importance(top_n=10)

pcr_model.print_best_k()
val_metrics = pcr_model.evaluate(X_val, y_val)
pcr_model.print_summary("PCR Validation", val_metrics)
test_metrics = pcr_model.evaluate(X_test, y_test)
pcr_model.print_summary("PCR Test", test_metrics)
pcr_model.print_feature_importance(top_n=10)

pls_model.print_best_k()
val_metrics = pls_model.evaluate(X_val, y_val)
pls_model.print_summary("PLS Validation", val_metrics)
test_metrics = pls_model.evaluate(X_test, y_test)
pls_model.print_summary("PLS Test", test_metrics)
pls_model.print_feature_importance(top_n=10)

glm.print_hyperparameters()
val_metrics = glm.evaluate(X_val, y_val)
glm.print_summary("GLM Validation", val_metrics)
test_metrics = glm.evaluate(X_test, y_test)
glm.print_summary("GLM Test", test_metrics)
glm.print_feature_importance()


nn_model.print_architecture()
val_metrics = nn_model.evaluate(X_val, y_val)
nn_model.print_summary("Neural Network Validation", val_metrics)
test_metrics = nn_model.evaluate(X_test, y_test)
nn_model.print_summary("Neural Network Test", test_metrics)
'''

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
