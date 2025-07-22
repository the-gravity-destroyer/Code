def train_elastic_net_model(X_train, y_train, X_test, y_test, alpha=1.0, l1_ratio=0.5):
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elastic_net_model.fit(X_train, y_train)

    y_pred = elastic_net_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"ElasticNet - Out-of-Sample R²: {r2:.4f}")
    print(f"ElasticNet - Out-of-Sample MSE: {mse:.4f}")

    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.title("ElasticNet: Predicted vs. Actual Returns")
    plt.show()

    return elastic_net_model, y_pred, r2, mse