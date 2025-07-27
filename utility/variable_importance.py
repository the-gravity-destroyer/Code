# variable_importance.py

import numpy as np
import pandas as pd

def drop_feature_importance(
    model,                # already trained BaseRegressor-Instance
    model_kwargs: dict,   # dict with __init__-arguments of the model (z.B. {'n_stocks':10, 'alpha':0.1, ...})
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    feature_names: list[str]
) -> pd.Series:
    """
    Calculates variable importance as a drop in out-of-sample R² when a feature column is omitted.

    Parameters
    ----------
    model : BaseRegressor
    An instance of the model already fitted with model.train(...).
    model_kwargs : dict
    The parameters used to build a new copy of the model,
    e.g., {'n_stocks': 10, 'alpha': 0.1, 'l1_ratio': 0.5}.
    X_train, X_val, X_test : np.ndarray, shape (T, p)
    y_train, y_val, y_test : np.ndarray, shape (T,)
    feature_names : list of str, length = p

    Returns
    -------
    pd.Series
    Index = feature_names, value = drop in R² (baseline_r2 − r2_ohne_feature).


    """
    # 1) Baseline-R² with all Features
    baseline_r2 = model.evaluate(X_test, y_test)['r2']

    drops = []
    p = X_train.shape[1]

    # 2) For each feature j, train a new model instance without this feature
    ModelClass = type(model)
    for j in range(p):
        # Data without column j
        X_tr = np.delete(X_train, j, axis=1)
        X_v  = np.delete(X_val,   j, axis=1)
        X_te = np.delete(X_test,  j, axis=1)

        # New instance with the same hyperparameters
        m = ModelClass(**model_kwargs)
        m.train(X_tr, y_train, X_v, y_val)
        r2_j = m.evaluate(X_te, y_test)['r2']

        # Importance = drop in R²
        drops.append(baseline_r2 - r2_j)

    return pd.Series(drops, index=feature_names)
