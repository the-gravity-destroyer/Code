# models/ordinary_least_squares_regression.py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_ols_model(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    n_stocks=10,
    plot=True
):
    """
    Trainiert ein OLS-Modell mit Standardisierung und liefert:
     - Zeitreihen-R² auf Validation und Test
     - Benchmark ('Zero-Return') R²
     - Cross-Sectional R² im Test-Set (monatlich über n_stocks Stocks)
     - MSE auf Validation und Test
     - standardisierte Koeffizienten nach Wichtigkeit
    Optional: Residual- und Scatter-Plots.
    """
    # 1) Pipeline definieren und fitten
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ols',    LinearRegression())
    ])
    pipeline.fit(X_train, y_train)

    # 2) Validation-Performance
    y_pred_val = pipeline.predict(X_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)
    r2_zero_val = 1 - np.sum((y_val - y_pred_val)**2) / np.sum(y_val**2)
    print(f"OLS - Validation R²:            {r2_val:.4f}")
    print(f"OLS - Validation MSE:           {mse_val:.4f}")
    print(f"OLS - Validation Zero-Return R²: {r2_zero_val:.4f}\n")

    # 3) Test-Performance
    y_pred_test = pipeline.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    r2_zero_test = 1 - np.sum((y_test - y_pred_test)**2) / np.sum(y_test**2)

    # Cross-sectional R² im Test-Set
    n_test = len(y_test)
    # Monatliche Indices basierend auf flachem Index
    months = np.arange(n_test) // n_stocks
    n_months_test = months.max() + 1

    # Mittelwerte je Monat
    y_bar    = np.array([y_test[months==m].mean()     for m in range(n_months_test)])
    yhat_bar = np.array([y_pred_test[months==m].mean() for m in range(n_months_test)])

    # Repliziere Mittelwerte über Stocks
    y_bar_rep    = np.repeat(y_bar,    [np.sum(months==m) for m in range(n_months_test)])
    yhat_bar_rep = np.repeat(yhat_bar, [np.sum(months==m) for m in range(n_months_test)])

    num_cs = np.sum(((y_test - y_bar_rep) - (y_pred_test - yhat_bar_rep))**2)
    den_cs = np.sum((y_test - y_bar_rep)**2)
    r2_cs = 1 - num_cs/den_cs

    print(f"OLS - Test R²:                   {r2_test:.4f}")
    print(f"OLS - Test MSE:                  {mse_test:.4f}")
    print(f"OLS - Test Zero-Return R²:       {r2_zero_test:.4f}")
    print(f"OLS - Test Cross-Sectional R²:   {r2_cs:.4f}\n")

    # 4) Standardisierte Koeffizienten (Feature Importance)
    coefs = pipeline.named_steps['ols'].coef_
    importance = np.abs(coefs)
    idx_sorted = np.argsort(-importance)
    print("Top-Features nach |standardisiertem Koeffizienten|:")
    for rank, idx in enumerate(idx_sorted[:10], 1):
        print(f"  {rank:>2}. Feature {idx:>2} → |coef| = {importance[idx]:.4f}")
    print()

    # 5) Plots (optional)
    if plot:
        residuals = y_test - y_pred_test
        plt.figure()
        plt.scatter(y_pred_test, residuals, alpha=0.6)
        plt.axhline(0, linestyle='--', color='gray')
        plt.xlabel("Predicted Returns")
        plt.ylabel("Residuals")
        plt.title("OLS: Residuals vs. Predicted")
        plt.show()

        plt.figure()
        plt.scatter(y_test, y_pred_test, alpha=0.6)
        mn, mx = y_test.min(), y_test.max()
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel("Actual Returns")
        plt.ylabel("Predicted Returns")
        plt.title("OLS: Predicted vs. Actual")
        plt.show()

    # 6) Rückgabe
    return pipeline, y_pred_test, r2_test, mse_test, r2_zero_test, r2_cs
