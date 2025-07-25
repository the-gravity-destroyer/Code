from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from models.linear_models.base_regressor import BaseRegressor
import numpy as np

class OLSModel(BaseRegressor):
    """OLS-Regressionsmodell mit Basisklasse-Unterstützung."""

    def __init__(self, n_stocks=None):
        super().__init__(n_stocks=n_stocks)

    def build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('ols',    LinearRegression())
        ])

    def get_standardized_coefficients(self):
        if not self.is_fitted:
            raise RuntimeError("Call train() first.")
        coefs = self.pipeline.named_steps['ols'].coef_
        importance = np.abs(coefs)
        idx_sorted = np.argsort(-importance)
        return importance, idx_sorted

    def print_feature_importance(self, top_n=10):
        importance, idx_sorted = self.get_standardized_coefficients()
        print("Top Features nach |standardisiertem Koeffizienten|:")
        for rank, idx in enumerate(idx_sorted[:top_n], 1):
            print(f"  {rank:>2}. Feature {idx:>2} → |coef| = {importance[idx]:.4f}")
        print()
