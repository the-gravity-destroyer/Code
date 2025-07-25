# models/principal_component_regression.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from models.linear_models.base_regressor import BaseRegressor

class PCRModel(BaseRegressor):
    def __init__(self, n_stocks=None, k_values=None):
        super().__init__(n_stocks=n_stocks)
        self.k_values = k_values if k_values is not None else None
        self.best_k = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            raise ValueError("X_val und y_val für Hyperparameter-Suche erforderlich.")
        n_features = X_train.shape[1]
        k_list = self.k_values or list(range(1, n_features + 1))
        best_mse = np.inf
        for k in k_list:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('pca',    PCA(n_components=k)),
                ('ols',    LinearRegression())
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

    def build_pipeline(self):
        if not self.is_fitted:
            raise RuntimeError("Call train() first to determine best_k.")
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=self.best_k)),
            ('ols',    LinearRegression())
        ])

    def print_best_k(self):
        print(f"PCR – beste Komponentenanzahl k = {self.best_k}")

    def get_feature_importance(self):
        if not self.is_fitted:
            raise RuntimeError("Call train() first.")
        pca = self.pipeline.named_steps['pca']
        beta = self.pipeline.named_steps['ols'].coef_
        feat_imp = np.abs(pca.components_.T @ beta)
        idx_sorted = np.argsort(-feat_imp)
        return feat_imp, idx_sorted

    def print_feature_importance(self, top_n=10):
        feat_imp, idx_sorted = self.get_feature_importance()
        print("PCR Top-Features nach |Loading×Coef|:")
        for rank, idx in enumerate(idx_sorted[:top_n], 1):
            print(f"  {rank:>2}. Feature {idx:>2} → Importance = {feat_imp[idx]:.4f}")
        print()

