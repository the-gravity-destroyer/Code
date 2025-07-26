# variable_importance.py

import numpy as np
import pandas as pd

def drop_feature_importance(
    model,                # bereits trainierte BaseRegressor-Instanz
    model_kwargs: dict,   # dict mit den __init__-Argumenten des Modells (z.B. {'n_stocks':10, 'alpha':0.1, ...})
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    feature_names: list[str]
) -> pd.Series:
    """
    Berechnet Variable Importance als Abfall in Out-of-Sample-R²,
    wenn eine Feature-Spalte weggelassen wird.

    Parameters
    ----------
    model : BaseRegressor
        Eine bereits mit model.train(...) fit-Instanz des Modells.
    model_kwargs : dict
        Die Parameter, mit denen man eine neue Kopie des Modells baut,
        z.B. {'n_stocks':10, 'alpha':0.1, 'l1_ratio':0.5}.
    X_train, X_val, X_test : np.ndarray, shape (T, p)
    y_train, y_val, y_test : np.ndarray, shape (T,)
    feature_names : list of str, Länge = p

    Returns
    -------
    pd.Series
        Index = feature_names, Wert = drop in R² (baseline_r2 − r2_ohne_feature).
    """
    # 1) Baseline-R² mit allen Features
    baseline_r2 = model.evaluate(X_test, y_test)['r2']

    drops = []
    p = X_train.shape[1]

    # 2) Für jedes Feature j eine neue Modell-Kopie trainieren ohne dieses Merkmal
    ModelClass = type(model)
    for j in range(p):
        # Daten ohne Spalte j
        X_tr = np.delete(X_train, j, axis=1)
        X_v  = np.delete(X_val,   j, axis=1)
        X_te = np.delete(X_test,  j, axis=1)

        # Neue Instanz mit denselben Hyperparametern
        m = ModelClass(**model_kwargs)
        m.train(X_tr, y_train, X_v, y_val)
        r2_j = m.evaluate(X_te, y_test)['r2']

        # Importance = Abfall in R²
        drops.append(baseline_r2 - r2_j)

    return pd.Series(drops, index=feature_names)
