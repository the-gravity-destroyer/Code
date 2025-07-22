def train_glm_model(X_train, y_train, X_test, y_test, n_components=2):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(X_train, y_train)

    y_pred = pls_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"PLS - Out-of-Sample R²: {r2:.4f}")
    print(f"PLS - Out-of-Sample MSE: {mse:.4f}")

    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.title("PLS: Predicted vs. Actual Returns")
    plt.show()

    return pls_model, y_pred, r2, mse