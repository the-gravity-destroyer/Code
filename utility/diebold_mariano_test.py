import numpy as np
from scipy.stats import norm

def diebold_mariano(y_true, y_pred1, y_pred2, h=1, loss='mse'):
    """
    Performs the Diebold–Mariano test for two forecast series.

    Parameters
    ----------
    y_true : array-like, shape (T,)
        Actual observations.
    y_pred1, y_pred2 : array-like, shape (T,)
        Predictions from the two models.
    h : int, default=1
        Forecast horizon. For h>1, a Newey–West correction with lag h-1 is used.
    loss : {'mse', 'mae'}, default='mse'
        Loss function: mean squared error or mean absolute error.

    Returns
    -------
    dm_stat : float
        Diebold–Mariano statistic.
    p_value : float
        Two-sided p-value under the standard normal distribution.
    """
    # Forecast-Errors
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    # Loss differences
    if loss == 'mse':
        d = e1**2 - e2**2
    elif loss == 'mae':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    T = len(d)
    mean_d = np.mean(d)

    # Newey–West-estimator for Var(mean_d) with Lag M=h-1
    M = max(h-1, 0)
    # Autokovariances gamma_k
    d_centered = d - mean_d
    gamma = [np.dot(d_centered[:T-k], d_centered[k:]) / T for k in range(M+1)]
    # Weights w_k: w0=1, wk = 1 - k/(M+1)
    weights = [1.0] + [(1.0 - k/(M+1)) for k in range(1, M+1)]
    var_d = gamma[0] + 2.0 * sum(weights[k] * gamma[k] for k in range(1, M+1))
    var_mean = var_d / T

    # DM-Statistic und p-Wert
    dm_stat = mean_d / np.sqrt(var_mean)
    p_value = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))

    return dm_stat, p_value
