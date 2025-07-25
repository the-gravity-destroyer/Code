from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from models.base_regressor import BaseRegressor


class GradientBoostingModel(BaseRegressor):
    """Gradient Boosting Regressor mit optionaler Hyperparameter-Suche."""
    def __init__(
        self,
        n_stocks=None,
        estimators=[100, 200],
        learning_rates=[0.1, 0.01],
        max_depths=[3, 5],
        random_state=42
    ):
        super().__init__(n_stocks=n_stocks)
        self.estimators    = estimators
        self.learning_rates= learning_rates
        self.max_depths    = max_depths
        self.random_state  = random_state
        self.best_params   = {}

    def build_pipeline(self):
        # Fallback: verwendet best_params, falls train() bereits lief
        return Pipeline([
            ('scaler', StandardScaler()),
            ('gbr',    GradientBoostingRegressor(
                         n_estimators=self.best_params.get('n_estimators', self.estimators[0]),
                         learning_rate=self.best_params.get('learning_rate', self.learning_rates[0]),
                         max_depth=self.best_params.get('max_depth', self.max_depths[0]),
                         random_state=self.random_state
                     ))
        ])

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            raise ValueError("X_val und y_val für Hyperparameter-Suche erforderlich.")
        best_mse = np.inf
        for n in self.estimators:
            for lr in self.learning_rates:
                for md in self.max_depths:
                    pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('gbr',    GradientBoostingRegressor(
                                     n_estimators=n,
                                     learning_rate=lr,
                                     max_depth=md,
                                     random_state=self.random_state
                                 ))
                    ])
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_val)
                    mse_val = mean_squared_error(y_val, y_pred)
                    if mse_val < best_mse:
                        best_mse = mse_val
                        self.pipeline  = pipe
                        self.best_params = {
                            'n_estimators': n,
                            'learning_rate': lr,
                            'max_depth': md
                        }
        self.is_fitted = True
        return self

    def print_hyperparameters(self):
        print("GradientBoosting – beste Hyperparameter:")
        for k, v in self.best_params.items():
            print(f"  {k:15s}: {v}")
        print()

    def get_feature_importance(self):
        if not self.is_fitted:
            raise RuntimeError("Call train() first.")
        imp = self.pipeline.named_steps['gbr'].feature_importances_
        idx_sorted = np.argsort(-imp)
        return imp, idx_sorted

    def print_feature_importance(self, top_n=10):
        imp, idx_sorted = self.get_feature_importance()
        print("GBR Top-Features nach importance_:")
        for rank, idx in enumerate(idx_sorted[:top_n], 1):
            print(f"  {rank:>2}. Feature {idx:>2} → importance = {imp[idx]:.4f}")
        print()
