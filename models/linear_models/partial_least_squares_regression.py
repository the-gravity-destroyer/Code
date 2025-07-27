# models/partial_least_squares_regression.py

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from models.base_regressor import BaseRegressor

class PLSModel(BaseRegressor):
    """Partial Least Squares Regression with component selection."""
    def __init__(self, n_stocks=None, n_components=None):
        super().__init__(n_stocks=n_stocks)
        self.n_components = n_components
        self.best_k = None

    def build_pipeline(self):
        # Fallback pipeline, will be replaced after train() with the best k
        k = self.n_components or 1
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pls',    PLSRegression(n_components=k))
        ])

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val are required for hyperparameter search.")
        n_features = X_train.shape[1]
        # Component range: if n_components is specified, use only this value
        ks = [self.n_components] if self.n_components else list(range(1, n_features + 1))
        best_mse = np.inf
        for k in ks:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('pls',    PLSRegression(n_components=k))
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            mse_val = mean_squared_error(y_val, y_pred)
            if mse_val < best_mse:
                best_mse = mse_val
                self.pipeline = pipe
                self.best_k = k
        self.is_fitted = True
        return self

    def print_best_k(self):
        print(f"PLS – best number of components k = {self.best_k}")

    def get_feature_importance(self):
        if not self.is_fitted:
            raise RuntimeError("Call train() first.")
        coefs = self.pipeline.named_steps['pls'].coef_.ravel()
        importance = np.abs(coefs)
        idx_sorted = np.argsort(-importance)
        return importance, idx_sorted

    def print_feature_importance(self, top_n=10):
        importance, idx_sorted = self.get_feature_importance()
        print("PLS top features by |coefficient|:")
        for rank, idx in enumerate(idx_sorted[:top_n], 1):
            print(f"  {rank:>2}. Feature {idx:>2} → |coef| = {importance[idx]:.4f}")
        print()

