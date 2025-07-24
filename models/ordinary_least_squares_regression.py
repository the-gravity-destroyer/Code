from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
def train_ols_model(X_train, y_train, X_val, y_val, X_test, y_test, plot=True):
    """
    Trainiert ein OLS-Modell innerhalb einer Pipeline mit Standardisierung.
    Gibt Performance auf Validierungs- und Test-Set aus und zeigt Residuals- und Scatter-Plots.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ols', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)

    # Validierungs-Performance
    y_pred_val = pipeline.predict(X_val)
    r2_val = r2_score(y_val, y_pred_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    print(f"OLS - Validation R²: {r2_val:.4f}")
    print(f"OLS - Validation MSE: {mse_val:.4f}")

    # Test-Performance
    y_pred_test = pipeline.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f"OLS - Test R²: {r2_test:.4f}")
    print(f"OLS - Test MSE: {mse_test:.4f}")

    if plot:
        # Residuen vs. Vorhersage
        residuals = y_test - y_pred_test
        plt.figure()
        plt.scatter(y_pred_test, residuals, alpha=0.6)
        plt.axhline(0, linestyle='--')
        plt.xlabel("Predicted Returns")
        plt.ylabel("Residuals")
        plt.title("OLS: Residuals vs. Predicted Returns")
        plt.show()

        # Predicted vs. Actual
        plt.figure()
        plt.scatter(y_test, y_pred_test, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual Returns")
        plt.ylabel("Predicted Returns")
        plt.title("OLS: Predicted vs. Actual Returns")
        plt.show()

    return pipeline, y_pred_test, r2_test, mse_test
