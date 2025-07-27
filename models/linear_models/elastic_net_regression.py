from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from models.base_regressor import BaseRegressor
import numpy as np


class ElasticNetModel(BaseRegressor):
    """Elastic Net Regression with manual hyperparameter search."""
    def __init__(self, n_stocks=None, alphas=None, l1_ratios=None):
        super().__init__(n_stocks=n_stocks)
        # Default-Grid
        self.alphas = alphas if alphas is not None else [0.01, 0.1, 1.0]
        self.l1_ratios = l1_ratios if l1_ratios is not None else [0.2, 0.5, 0.8]
        self.base_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('en',     ElasticNet(max_iter=10_000, tol=1e-4))
        ])
        self.best_params = {}

    def build_pipeline(self):
        # Not used directly, since train() sets the best pipeline
        return self.base_pipe

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Performs grid search on the validation set and stores the best model.
        """
        if X_val is None or y_val is None:
            raise ValueError("X_val und y_val für Hyperparameter-Search required.")
        best_mse = np.inf
        for alpha in self.alphas:
            for l1 in self.l1_ratios:
                pipe = clone(self.base_pipe)
                pipe.set_params(en__alpha=alpha, en__l1_ratio=l1)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_val)
                mse_val = mean_squared_error(y_val, y_pred)
                if mse_val < best_mse:
                    best_mse = mse_val
                    self.pipeline = pipe
                    self.best_params = {'alpha': alpha, 'l1_ratio': l1}
        self.is_fitted = True
        return self

    def print_hyperparameters(self):
        """Prints the chosen hyperparameters and validation MSE."""
        print("ElasticNet – best Hyperparameters:")
        print(f"  alpha = {self.best_params['alpha']}, l1_ratio = {self.best_params['l1_ratio']}")

    def get_standardized_coefficients(self):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call train() first.")
        coefs = self.pipeline.named_steps['en'].coef_
        importance = np.abs(coefs)
        idx_sorted = np.argsort(-importance)
        return importance, idx_sorted

    def print_feature_importance(self, top_n=10):
        importance, idx_sorted = self.get_standardized_coefficients()
        print("ElasticNet top features by |standardized coefficient|:")
        for rank, idx in enumerate(idx_sorted[:top_n], 1):
            print(f"  {rank:>2}. Feature {idx:>2} → |coef| = {importance[idx]:.4f}")
        print()