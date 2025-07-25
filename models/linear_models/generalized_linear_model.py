# models/generalized_linear_model.py

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from models.base_regressor import BaseRegressor

class GLMModel(BaseRegressor):
    """Generalized Linear Model: Splines + ElasticNet-Regularisierung."""
    def __init__(self, n_stocks=None, knot_options=None, alphas=None, l1_ratios=None):
        super().__init__(n_stocks=n_stocks)
        self.knot_options = knot_options if knot_options is not None else [3,5,7]
        self.alphas       = alphas         if alphas       is not None else [0.01, 0.1, 1.0]
        self.l1_ratios    = l1_ratios      if l1_ratios    is not None else [0.2, 0.5, 0.8]
        self.best_params  = {}
        self.p_orig       = None  # Anzahl Original-Features

    def build_pipeline(self, n_knots):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('spline', SplineTransformer(degree=3, n_knots=n_knots, include_bias=False)),
            ('en',     ElasticNet(max_iter=10_000, tol=1e-4,
                                 alpha=self.best_params.get('alpha', 1.0),
                                 l1_ratio=self.best_params.get('l1_ratio', 0.5)))
        ])

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            raise ValueError("X_val und y_val für Hyperparameter-Suche erforderlich.")
        # Original Feature Count speichern
        self.p_orig = X_train.shape[1]
        best_mse = np.inf

        # Schleife über Splines, alpha und l1_ratio
        for k in self.knot_options:
            for alpha in self.alphas:
                for l1 in self.l1_ratios:
                    pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('spline', SplineTransformer(degree=3, n_knots=k, include_bias=False)),
                        ('en',     ElasticNet(max_iter=10_000, tol=1e-4,
                                             alpha=alpha, l1_ratio=l1))
                    ])
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_val)
                    mse_val = mean_squared_error(y_val, y_pred)
                    if mse_val < best_mse:
                        best_mse = mse_val
                        self.pipeline    = pipe
                        self.best_params = {
                            'n_knots': k,
                            'alpha':   alpha,
                            'l1_ratio':l1
                        }
        self.is_fitted = True
        return self

    def print_hyperparameters(self):
        print("GLM Model – beste Hyperparameter:")
        print(f"  n_knots  = {self.best_params['n_knots']}")
        print(f"  alpha    = {self.best_params['alpha']}")
        print(f"  l1_ratio = {self.best_params['l1_ratio']}")

    def get_feature_importance(self):
        if not self.is_fitted:
            raise RuntimeError("Call train() first.")
        # absolute Koeffizienten je Spline-Basis
        coefs = self.pipeline.named_steps['en'].coef_
        # Aggregation je Original-Feature
        p = self.p_orig
        # Anzahl Splines pro Feature = len(coefs) // p
        n_basis = coefs.shape[0] // p
        importances = [
            np.sum(np.abs(coefs[i*n_basis:(i+1)*n_basis]))
            for i in range(p)
        ]
        idx_sorted = np.argsort(-np.array(importances))
        return importances, idx_sorted

    def print_feature_importance(self, top_n=10):
        imps, idx_sorted = self.get_feature_importance()
        print("GLM Top-Features (aggregierte |Spline-Koeffizienten|):")
        for rank, idx in enumerate(idx_sorted[:top_n], 1):
            print(f"  {rank:>2}. Feature {idx:>2} → Importance = {imps[idx]:.4f}")
        print()
