# models/elastic_net_regression.py

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_elastic_net_model(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    n_stocks=10,
    plot=False
):
    """
    Trainiert ein ElasticNet-Modell mit manueller Hyperparameter-Suche auf dem Validation-Set.
    Liefert:
     - Bestes alpha & l1_ratio nach MSE auf Validation
     - Zeitreihen-R², Zero-Return R², Cross-Sectional R² im Test-Set
     - MSE auf Validation und Test
     - standardisierte Koeffizienten
    Optional: Residual- und Scatter-Plots.
    """
    # 1) Pipeline-Grundgerüst
    base_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('en',     ElasticNet(max_iter=10_000, tol=1e-4))
    ])

    # 2) Hyperparameter-Gitter
    alphas     = [0.01, 0.1, 1.0]
    l1_ratios  = [0.2, 0.5, 0.8]

    best_mse = np.inf
    best_pipe = None
    best_params = {}

    # 3) Manueller Grid-Search
    for alpha in alphas:
        for l1 in l1_ratios:
            pipe = clone(base_pipe)
            pipe.set_params(en__alpha=alpha, en__l1_ratio=l1)
            pipe.fit(X_train, y_train)
            y_val_pred = pipe.predict(X_val)
            mse_val = mean_squared_error(y_val, y_val_pred)
            if mse_val < best_mse:
                best_mse = mse_val
                best_pipe = pipe
                best_params = {'alpha': alpha, 'l1_ratio': l1}

    # Ergebnis der Suche ausgeben
    print("ElasticNet – Beste Hyperparameter (nach Validation MSE):")
    print(f"  alpha = {best_params['alpha']}, l1_ratio = {best_params['l1_ratio']}")
    print(f"  Validation MSE = {best_mse:.4f}\n")

    # 4) Validierungsmetriken mit dem besten Modell
    y_val_pred = best_pipe.predict(X_val)
    r2_val     = r2_score(y_val, y_val_pred)
    r2_zero_val= 1 - np.sum((y_val - y_val_pred)**2) / np.sum(y_val**2)
    print(f"ElasticNet - Validation R²:            {r2_val:.4f}")
    print(f"ElasticNet - Validation Zero-Return R²: {r2_zero_val:.4f}\n")

    # 5) Test-Metriken
    y_test_pred = best_pipe.predict(X_test)
    mse_test    = mean_squared_error(y_test, y_test_pred)
    r2_test     = r2_score(y_test, y_test_pred)
    r2_zero_test= 1 - np.sum((y_test - y_test_pred)**2) / np.sum(y_test**2)

    # Cross-sectional R² im Test-Set
    n_test = len(y_test)
    months = np.arange(n_test) // n_stocks
    n_months_test = months.max() + 1
    y_bar     = np.array([y_test[months==m].mean()     for m in range(n_months_test)])
    yhat_bar  = np.array([y_test_pred[months==m].mean() for m in range(n_months_test)])
    # Replizieren
    counts    = np.array([np.sum(months==m) for m in range(n_months_test)])
    y_bar_rep = np.repeat(y_bar,    counts)
    yhat_rep  = np.repeat(yhat_bar, counts)
    num_cs    = np.sum(((y_test - y_bar_rep) - (y_test_pred - yhat_rep))**2)
    den_cs    = np.sum((y_test - y_bar_rep)**2)
    r2_cs      = 1 - num_cs/den_cs

    print(f"ElasticNet - Test R²:                   {r2_test:.4f}")
    print(f"ElasticNet - Test Zero-Return R²:       {r2_zero_test:.4f}")
    print(f"ElasticNet - Test Cross-Sectional R²:   {r2_cs:.4f}")
    print(f"ElasticNet - Test MSE:                  {mse_test:.4f}\n")

    # 6) Standardisierte Koeffizienten (Feature Importance)
    coefs = best_pipe.named_steps['en'].coef_
    importance = np.abs(coefs)
    idx_sorted = np.argsort(-importance)
    print("ElasticNet Top-Features nach |standardisiertem Koeffizienten|:")
    for rank, idx in enumerate(idx_sorted[:10], 1):
        print(f"  {rank:>2}. Feature {idx:>2} → |coef| = {importance[idx]:.4f}")
    print()

    # 7) Plots (optional)
    if plot:
        residuals = y_test - y_test_pred
        plt.figure()
        plt.scatter(y_test_pred, residuals, alpha=0.6)
        plt.axhline(0, linestyle='--', color='gray')
        plt.xlabel("Predicted Returns")
        plt.ylabel("Residuals")
        plt.title("ElasticNet: Residuals vs. Predicted")
        plt.show()

        plt.figure()
        plt.scatter(y_test, y_test_pred, alpha=0.6)
        mn, mx = y_test.min(), y_test.max()
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel("Actual Returns")
        plt.ylabel("Predicted Returns")
        plt.title("ElasticNet: Predicted vs. Actual")
        plt.show()

    # 8) Rückgabe
    return best_pipe, y_test_pred, r2_test, mse_test, r2_zero_test, r2_cs
