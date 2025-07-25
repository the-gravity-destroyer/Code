# models/principal_component_regression.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_pcr_model(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    n_stocks=10,
    plot=False
):
    """
    Trainiert ein Principal Component Regression (PCR)-Modell:
     - Wählt Anzahl Komponenten k anhand des Validation-MSE
     - Liefert Zeitreihen-R², Zero-Return-R², Cross-Sectional-R², MSE auf Val & Test
     - Berechnet Feature-Importance via PCA-Loadings * Regressionsgewichte
    """
    # Feature-Anzahl
    n_features = X_train.shape[1]

    # Suchbereich für k
    k_values = list(range(1, n_features + 1))

    best_mse = np.inf
    best_k = None
    best_pipe = None

    # Grid-Search über k
    for k in k_values:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=k)),
            ('ols',    LinearRegression())
        ])
        pipe.fit(X_train, y_train)
        y_val_pred = pipe.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        if mse_val < best_mse:
            best_mse = mse_val
            best_k = k
            best_pipe = pipe

    print(f"PCR – beste Komponentenanzahl k = {best_k} (Validation MSE = {best_mse:.4f})\n")

    # Validation-Metriken mit bestem Modell
    y_val_pred = best_pipe.predict(X_val)
    r2_val     = r2_score(y_val, y_val_pred)
    r2_zero_val= 1 - np.sum((y_val - y_val_pred)**2) / np.sum(y_val**2)
    mse_val    = mean_squared_error(y_val, y_val_pred)
    print(f"PCR - Validation R²:            {r2_val:.4f}")
    print(f"PCR - Validation Zero-Return R²: {r2_zero_val:.4f}")
    print(f"PCR - Validation MSE:           {mse_val:.4f}\n")

    # Test-Metriken
    y_test_pred = best_pipe.predict(X_test)
    mse_test    = mean_squared_error(y_test, y_test_pred)
    r2_test     = r2_score(y_test, y_test_pred)
    r2_zero_test= 1 - np.sum((y_test - y_test_pred)**2) / np.sum(y_test**2)

    # Cross-Sectional R² im Test-Set
    n_test = len(y_test)
    months = np.arange(n_test) // n_stocks
    n_months_test = months.max() + 1
    y_bar    = np.array([y_test[months==m].mean()     for m in range(n_months_test)])
    yhat_bar = np.array([y_test_pred[months==m].mean() for m in range(n_months_test)])
    counts   = np.array([np.sum(months==m) for m in range(n_months_test)])
    y_bar_rep    = np.repeat(y_bar,    counts)
    yhat_bar_rep = np.repeat(yhat_bar, counts)
    num_cs = np.sum(((y_test - y_bar_rep) - (y_test_pred - yhat_bar_rep))**2)
    den_cs = np.sum((y_test - y_bar_rep)**2)
    r2_cs = 1 - num_cs/den_cs

    print(f"PCR - Test R²:                   {r2_test:.4f}")
    print(f"PCR - Test Zero-Return R²:       {r2_zero_test:.4f}")
    print(f"PCR - Test Cross-Sectional R²:   {r2_cs:.4f}")
    print(f"PCR - Test MSE:                  {mse_test:.4f}\n")

    # Feature Importance: PC-Loadings × Regressionskoeffizienten
    pca = best_pipe.named_steps['pca']
    beta = best_pipe.named_steps['ols'].coef_  # (k,)
    # components_.shape = (k, n_features)
    feat_importance = np.abs(pca.components_.T @ beta)  # (n_features,)
    idx_sorted = np.argsort(-feat_importance)
    print("PCR Top-Features nach |Loading×Coef|:")
    for rank, idx in enumerate(idx_sorted[:10], 1):
        print(f"  {rank:>2}. Feature {idx:>2} → Importance = {feat_importance[idx]:.4f}")
    print()

    # Optional: Plots
    if plot:
        residuals = y_test - y_test_pred
        plt.figure()
        plt.scatter(y_test_pred, residuals, alpha=0.6)
        plt.axhline(0, linestyle='--', color='gray')
        plt.xlabel("Predicted Returns")
        plt.ylabel("Residuals")
        plt.title(f"PCR (k={best_k}): Residuals vs. Predicted")
        plt.show()

        plt.figure()
        plt.scatter(y_test, y_test_pred, alpha=0.6)
        mn, mx = y_test.min(), y_test.max()
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel("Actual Returns")
        plt.ylabel("Predicted Returns")
        plt.title(f"PCR (k={best_k}): Predicted vs. Actual")
        plt.show()

    # Rückgabe
    return best_pipe, best_k, y_test_pred, r2_test, mse_test, r2_zero_test, r2_cs
