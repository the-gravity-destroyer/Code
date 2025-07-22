from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_ols_model(X_train, y_train, X_test, y_test, plot=True):
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)

    y_pred = ols_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"OLS - Out-of-Sample R²: {r2:.4f}")
    print(f"OLS - Out-of-Sample MSE: {mse:.4f}")

    if plot:
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel("Actual Returns")
        plt.ylabel("Predicted Returns")
        plt.title("OLS: Predicted vs. Actual Returns")
        plt.show()

    return ols_model, y_pred, r2, mse
