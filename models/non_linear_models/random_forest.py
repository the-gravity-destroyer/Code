# models/random_forest_regression.py

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from models.base_regressor import BaseRegressor

class RandomForestModel(BaseRegressor):
    """Random Forest regressor with optional hyperparameter search."""
    def __init__(
        self,
        n_stocks=None,
        n_estimators=[100, 200, 500],
        max_depths=[None, 5, 10],
        min_samples_leaf=[1, 5],
        random_state=42
    ):
        super().__init__(n_stocks=n_stocks)
        self.n_estimators     = n_estimators
        self.max_depths       = max_depths
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state
        self.best_params      = {}

    def build_pipeline(self):
        # Fallback pipeline with the best parameters (after train())
        return Pipeline([
            ('scaler', StandardScaler()),
            ('rf',     RandomForestRegressor(
                          n_estimators=self.best_params.get('n_estimators', self.n_estimators[0]),
                          max_depth=self.best_params.get('max_depth', self.max_depths[0]),
                          min_samples_leaf=self.best_params.get('min_samples_leaf', self.min_samples_leaf[0]),
                          random_state=self.random_state,
                          n_jobs=-1
                      ))
        ])

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val are required for hyperparameter search.")
        best_mse = np.inf

        # Grid-Search über Parameter
        for n in self.n_estimators:
            for d in self.max_depths:
                for leaf in self.min_samples_leaf:
                    pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('rf',     RandomForestRegressor(
                                      n_estimators=n,
                                      max_depth=d,
                                      min_samples_leaf=leaf,
                                      random_state=self.random_state,
                                      n_jobs=-1
                                  ))
                    ])
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_val)
                    mse_val = ((y_val - y_pred) ** 2).mean()
                    if mse_val < best_mse:
                        best_mse     = mse_val
                        self.pipeline    = pipe
                        self.best_params  = {
                            'n_estimators':   n,
                            'max_depth':      d,
                            'min_samples_leaf': leaf
                        }
        self.is_fitted = True
        return self

    def print_hyperparameters(self):
        print("RandomForest – best hyperparameters:")
        for k, v in self.best_params.items():
            print(f"  {k:20s}: {v}")
        print()

    def get_feature_importance(self):
        if not self.is_fitted:
            raise RuntimeError("Call train() first.")
        imp = self.pipeline.named_steps['rf'].feature_importances_
        idx_sorted = np.argsort(-imp)
        return imp, idx_sorted

    def print_feature_importance(self, top_n=10):
        imp, idx_sorted = self.get_feature_importance()
        print("RF top features according to feature_importances_:")
        for rank, idx in enumerate(idx_sorted[:top_n], 1):
            print(f"  {rank:>2}. Feature {idx:>2} → importance = {imp[idx]:.4f}")
        print()
