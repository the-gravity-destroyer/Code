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
    stock_characteristics, macro_factors, n_stocks, n_months = generate_synthetic_data()

    zi_t = []
    ri_t = []

    for t in range(1, n_months):
        for i in range(n_stocks):
            stock_feats = stock_characteristics[i, t, :]
            macro_feats = macro_factors[t, :]
            features = np.concatenate([stock_feats, macro_feats])
            zi_t.append(features)

            ret = np.dot(stock_feats, np.array([0.2, -0.1, 0.3, 0.1, 0.05])) + \
                  np.dot(macro_feats, np.array([0.3, -0.2, 0.1]))
            ri_t.append(ret)

    zi_t = np.array(zi_t)
    ri_t = np.array(ri_t)

    X_train_val, X_test, y_train_val, y_test = train_test_split(zi_t, ri_t, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2857, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


