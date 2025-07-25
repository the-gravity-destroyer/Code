from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class BaseRegressor(ABC):
    """
    Abstrakte Basisklasse für alle Regressoren.
    Stellt Standard-Interface für Training, Vorhersage, Evaluation und Plotting bereit.
    """

    def __init__(self, n_stocks=None):
        self.pipeline = None
        self.n_stocks = n_stocks
        self.is_fitted = False

    @abstractmethod
    def build_pipeline(self):
        """Muss eine sklearn-Pipeline oder ähnliches zurückgeben."""
        pass

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Trainiert die Pipeline nur mit Trainingsdaten."""
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Gibt Vorhersagen zurück. Muss nach train() aufgerufen werden."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call train() first.")
        return self.pipeline.predict(X)

    def evaluate(self, X, y):
        """
        Berechnet MSE, R2, Zero-Return R2 und optional Cross-Sectional R2, falls n_stocks vorliegt.
        """
        y_pred = self.predict(X)
        mse    = mean_squared_error(y, y_pred)
        r2     = r2_score(y, y_pred)
        r2_zero= 1 - np.sum((y - y_pred)**2) / np.sum(y**2)
        results = {'mse': mse, 'r2': r2, 'r2_zero': r2_zero}
        if self.n_stocks:
            n = len(y)
            months = np.arange(n) // self.n_stocks
            n_months = months.max() + 1
            y_bar    = np.array([y[months==m].mean()     for m in range(n_months)])
            yhat_bar = np.array([y_pred[months==m].mean() for m in range(n_months)])
            counts   = np.array([np.sum(months==m) for m in range(n_months)])
            y_bar_rep    = np.repeat(y_bar,    counts)
            yhat_bar_rep = np.repeat(yhat_bar, counts)
            num_cs = np.sum(((y - y_bar_rep) - (y_pred - yhat_bar_rep))**2)
            den_cs = np.sum((y - y_bar_rep)**2)
            results['r2_cs'] = 1 - num_cs/den_cs
        return results

    def print_summary(self, title, metrics):
        """Schöner Konsolen-Output der Kennzahlen."""
        print(f"=== {title} ===")
        for k, v in metrics.items():
            print(f"{k:10s}: {v:.4f}")
        print()

    def plot_diagnostics(self, X_test, y_test):
        """Residual- und Scatterplots für Test-Daten."""
        y_pred = self.predict(X_test)
        residuals = y_test - y_pred
        # Residuals vs Predicted
        plt.figure()
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, linestyle='--', color='gray')
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted")
        plt.show()
        # Actual vs Predicted
        plt.figure()
        plt.scatter(y_test, y_pred, alpha=0.6)
        mn, mx = y_test.min(), y_test.max()
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.show()
