import numpy as np  # Import NumPy for numerical operations
from sklearn.model_selection import train_test_split


def generate_synthetic_data():
    n_stocks = 10
    n_months = 24
    n_characteristics = 5
    n_macro_factors = 3

    np.random.seed(42)
    stock_characteristics = np.random.rand(n_stocks, n_months, n_characteristics)
    macro_factors = np.random.rand(n_months, n_macro_factors)
    return stock_characteristics, macro_factors, n_stocks, n_months


def split_data():
    """
    Generates synthetic data and splits it in a time-series fashion:
      - Features and returns are lagged by one period
      - First 70% of observations: Training + Validation (80/20 split)
      - Last 30% of observations: Test
    """
    stock_characteristics, macro_factors, n_stocks, n_months = generate_synthetic_data()

    features = []
    returns = []

    # Lagged features: für Retoure in Periode t nur Daten bis t-1 verwenden
    for t in range(1, n_months):
        for i in range(n_stocks):
            stock_prev = stock_characteristics[i, t-1, :]
            macro_prev = macro_factors[t-1, :]
            x_t = np.concatenate([stock_prev, macro_prev])
            # Synthetic Return ebenfalls basierend auf den gelaggten Merkmalen
            ret_t = (stock_prev.dot(np.array([0.2, -0.1, 0.3, 0.1, 0.05])) +
                     macro_prev.dot(np.array([0.3, -0.2, 0.1])))

            features.append(x_t)
            returns.append(ret_t)

    X = np.array(features)
    y = np.array(returns)

    n_total = len(y)
    test_frac = 0.3
    val_frac = 0.2

    n_test = int(n_total * test_frac)
    n_train_val = n_total - n_test
    n_val = int(n_train_val * val_frac)
    n_train = n_train_val - n_val

    X_train = X[:n_train]
    X_val   = X[n_train:n_train + n_val]
    X_test  = X[n_train + n_val:]

    y_train = y[:n_train]
    y_val   = y[n_train:n_train + n_val]
    y_test  = y[n_train + n_val:]

    return X_train, X_val, X_test, y_train, y_val, y_test

