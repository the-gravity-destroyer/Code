import numpy as np
from scipy.stats import norm

def diebold_mariano(y_true, y_pred1, y_pred2, h=1, loss='mse'):
    """
    Führt den Diebold–Mariano-Test für zwei Forecast-Serien durch.

    Parameter
    ----------
    y_true : array-like, Form (T,)
        Tatsächliche Beobachtungen.
    y_pred1, y_pred2 : array-like, Form (T,)
        Vorhersagen der beiden Modelle.
    h : int, default=1
        Vorhersagehorizont. Für h>1 wird ein Newey–West-Korrektur mit Lag h-1 verwendet.
    loss : {'mse', 'mae'}, default='mse'
        Verlustfunktion: mittlere quadratische oder absolute Fehler.

    Returns
    -------
    dm_stat : float
        Diebold–Mariano-Statistik.
    p_value : float
        Zwei-seitiger p-Wert unter Standardnormalverteilung.
    """
    # Forecast-Errors
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    # Verlustdifferenzen
    if loss == 'mse':
        d = e1**2 - e2**2
    elif loss == 'mae':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    T = len(d)
    mean_d = np.mean(d)

    # Newey–West-Schätzer für Var(mean_d) mit Lag M=h-1
    M = max(h-1, 0)
    # Autokovarianzen gamma_k
    d_centered = d - mean_d
    gamma = [np.dot(d_centered[:T-k], d_centered[k:]) / T for k in range(M+1)]
    # Gewichte w_k: w0=1, wk = 1 - k/(M+1)
    weights = [1.0] + [(1.0 - k/(M+1)) for k in range(1, M+1)]
    var_d = gamma[0] + 2.0 * sum(weights[k] * gamma[k] for k in range(1, M+1))
    var_mean = var_d / T

    # DM-Statistik und p-Wert
    dm_stat = mean_d / np.sqrt(var_mean)
    p_value = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))

    return dm_stat, p_value
