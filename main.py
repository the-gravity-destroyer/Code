# Problem Set: Multi-Model Financial Prediction and Risk Analysis
# Student Exercise Manual
# Please complete the code
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utility.variable_importance import drop_feature_importance
from utility.prediction_wrapper import PredictionWrapper
from utility.diebold_mariano_test import diebold_mariano
from utility.decile_portfolio import DecilePortfolioAnalysis


from models.linear_models.ordinary_least_squares_regression import OLSModel
from models.linear_models.elastic_net_regression import ElasticNetModel
from models.linear_models.principal_component_regression import PCRModel
from models.linear_models.partial_least_squares_regression import PLSModel
from models.linear_models.generalized_linear_model import GLMModel
from models.non_linear_models.neural_network import NeuralNetworkModel
from models.non_linear_models.gradient_boosting import GradientBoostingModel
from models.non_linear_models.random_forest import RandomForestModel

from data_processing import split_data

def main():
    """
    Main function to run the multi-model financial prediction and risk analysis.
    This function orchestrates the data generation, model training, evaluation,
    and visualization of results.
    """
    # Part 1: Data Generation and Preprocessing
    # Part 2: Model Training Section
    # Part 3: Prediction Wrappers
    # Part 4: Full-Sample Time Series Plotsim
    # Part 5: Out-of-Sample R² Results Table
    # Part 6: Diebold-Mariano Test Statistics
    # Part 7: Variable Importance Calculations & Heatmaps
    # Part 8: Auxiliary Functions and Decile Portfolio Analysis

    # -------------------------------
    # Part 1 & 2: Data Generation and Preprocessing
    # -------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_data()

    # -------------------------------
    # Part 3: Model Training Section
    # -------------------------------

    # --- OLS Model Training and Evaluation ---
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

    # --- Gradient Boosting Regressor ---
    gradient_boosting_model = GradientBoostingModel(n_stocks=10)
    gradient_boosting_model.train(X_train, y_train, X_val, y_val)

    # --- Random Forest Regressor ---
    random_forest_model = RandomForestModel(n_stocks=10)
    random_forest_model.train(X_train, y_train, X_val, y_val)

    # -------------------------------
    # Part 4: Prediction Wrappers
    # -------------------------------

    # collect all models in dictionary
    models = {
        'OLS':                ols_model,
        'ElasticNet':         elastic_net_model,
        'PCR':                pcr_model,
        'PLS':                pls_model,
        'GLM':                glm,
        'NeuralNetwork':      nn_model,
        'GradientBoosting':   gradient_boosting_model,
        'RandomForest':       random_forest_model
    }

    # instantiate the wrapper with the trained models
    wrapper = PredictionWrapper(models)

    # Collect predictions for all models on the test set
    df_preds = wrapper.predict(X_test)
    print(df_preds.head())   # zeigt die ersten Zeilen mit 8 Spalten

    # Out-of-Sample R² for all models in a loop
    for name in df_preds.columns:
        y_pred = df_preds[name].values
        metrics = models[name].evaluate(X_test, y_test)
        models[name].print_summary(f"{name} Test", metrics)

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
    gradient_boosting_model.plot_diagnostics(X_test, y_test)
    random_forest_model.plot_diagnostics(X_test, y_test)
    '''


    # -------------------------------
    # Part 6: Out-of-Sample R² Results Table - to evaluate model performance
    # -------------------------------
    val_metrics = ols_model.evaluate(X_val, y_val)
    ols_model.print_summary("OLS Validation", val_metrics)
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

    gradient_boosting_model.print_hyperparameters()
    val_metrics = gradient_boosting_model.evaluate(X_val, y_val)
    gradient_boosting_model.print_summary("Gradient Boosting Validation", val_metrics)
    test_metrics = gradient_boosting_model.evaluate(X_test, y_test)
    gradient_boosting_model.print_summary("Gradient Boosting Test", test_metrics)
    gradient_boosting_model.print_feature_importance(top_n=10)

    val_metrics = random_forest_model.evaluate(X_val, y_val)
    random_forest_model.print_summary("Random Forest Validation", val_metrics)
    test_metrics = random_forest_model.evaluate(X_test, y_test)
    random_forest_model.print_summary("Random Forest Test", test_metrics)
    random_forest_model.print_feature_importance(top_n=10)
    random_forest_model.print_hyperparameters()

    # -------------------------------
    # Part 7: Diebold-Mariano Test Statistics - to compare model predictions
    # -------------------------------
    from utility.diebold_mariano_test import diebold_mariano
    import numpy as np


    # y_true, y_pred_ols, y_pred_en jeweils aus deinen Modellen

    # Nachdem Du alle Modelle trainiert und ihre Vorhersagen erzeugt hast:


    # 1) Vorhersagen auf dem Test-Set
    y_pred_ols   = ols_model.predict(X_test)
    y_pred_en    = elastic_net_model.predict(X_test)
    y_pred_pcr   = pcr_model.predict(X_test)
    y_pred_pls   = pls_model.predict(X_test)
    y_pred_glm   = glm.predict(X_test)
    y_pred_rf    = random_forest_model.predict(X_test)
    y_pred_gb    = gradient_boosting_model.predict(X_test)
    y_pred_nn    = nn_model.predict(X_test)

    # 2) Ensembles
    linear_preds    = np.vstack([y_pred_ols, y_pred_en, y_pred_pcr, y_pred_pls, y_pred_glm]).mean(axis=0)
    nonlin_preds    = np.vstack([y_pred_rf, y_pred_gb, y_pred_nn]).mean(axis=0)

    # 3) Define DM comparisons
    comparisons = [
        ("OLS",             "ElasticNet",    y_pred_ols, linear_preds := y_pred_en),    # 1
        ("OLS",             "PCR",           y_pred_ols, y_pred_pcr),                   # 2
        ("OLS",             "PLS",           y_pred_ols, y_pred_pls),                   # 3
        ("OLS",             "GLM",           y_pred_ols, y_pred_glm),                   # 4
        ("OLS",             "RandomForest",  y_pred_ols, y_pred_rf),                    # 5
        ("OLS",             "GradientBoosting", y_pred_ols, y_pred_gb),                 # 6
        ("OLS",             "NeuralNetwork", y_pred_ols, y_pred_nn),                   # 7
        ("LinearEnsemble",  "NonlinearEnsemble", linear_preds, nonlin_preds),           # 8
        ("RandomForest",    "ElasticNet",    y_pred_rf,    y_pred_en),                 # 9
        ("RandomForest",    "NeuralNetwork", y_pred_rf,    y_pred_nn),                 # 10
        ("ElasticNet",      "NeuralNetwork", y_pred_en,    y_pred_nn),                 # 11
    ]

    # 4) Ausführen und ausgeben
    for name1, name2, pred1, pred2 in comparisons:
        dm_stat, p_val = diebold_mariano(y_test, pred1, pred2, h=1, loss='mse')
        print(f"{name1:20s} vs. {name2:20s} → DM = {dm_stat:7.3f}, p-value = {p_val:7.3f}")

    dm_stat, p_val = diebold_mariano(y_test, y_pred_ols, y_pred_en, h=1, loss='mse')
    print(f"DM-Statistic: {dm_stat:.3f}, p-value: {p_val:.3f}")

    # -------------------------------
    # Part 8: Variable Importance Calculations & Heatmaps - to understand feature importance ( to see which features are more important) TODO: Define a function to compute variable importance based on the drop in R² when a feature is removed
    # -------------------------------

    # 1) Define the feature names based on the training data
    feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]

    # 2) Dictionary for all models and their __init__ arguments
    models_info = {
        "OLS":               (ols_model,              {"n_stocks":10}),
        "ElasticNet":        (elastic_net_model,      {"n_stocks":10}),
        "PCR":               (pcr_model,              {"n_stocks":10}),
        "PLS":               (pls_model,              {"n_stocks":10}),
        "GLM":               (glm,                    {"n_stocks":10}),
        "NeuralNetwork":     (nn_model,               {"n_stocks":10}),
        "GradientBoosting":  (gradient_boosting_model,{"n_stocks":10}),
        "RandomForest":      (random_forest_model,    {"n_stocks":10})
    }

    # 3) Drop-in-R² for every model
    vi_dict = {}
    for name, (model_inst, init_args) in models_info.items():
        vi = drop_feature_importance(
            model_inst,
            init_args,
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            feature_names
        )
        vi_dict[name] = vi

    # 4) Concatenate in DataFrame
    df_vi = pd.concat(vi_dict, axis=1)

    # 5) Plotting Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_vi, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Variable Importance (Drop in R²) Across Models")
    plt.xlabel("Model")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


    # -------------------------------
    # Part 9: Auxiliary Functions and Decile Portfolio Analysis - to analyze model performance across deciles - to compare predicted vs actual  sharpe ratios
    # -------------------------------
    # 1) Instantiate with the number of stocks per period
    decile_analyzer = DecilePortfolioAnalysis(n_stocks=10)

    # 2) Dictionary with all predictions
    predictions = {
        "OLS":               y_pred_ols,
        "ElasticNet":        y_pred_en,
        "PCR":               y_pred_pcr,
        "PLS":               y_pred_pls,
        "GLM":               y_pred_glm,
        "NeuralNetwork":     y_pred_nn,
        "GradientBoosting":  y_pred_gb,
        "RandomForest":      y_pred_rf
    }

    # 3) Compute decile returns and plot Sharpe ratios for each model
    for model_name, y_pred in predictions.items():
        df_deciles = decile_analyzer.compute_decile_returns(y_pred, y_test)
        decile_analyzer.plot_sharpe(
            df_deciles,
            title=f"{model_name} Decile Sharpe Ratios"
        )


if __name__ == "__main__":
    main()