def train_pcr_model():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    def pcr(X_train, y_train, X_test, y_test, n_components):
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_pca, y_train)

        y_pred = model.predict(X_test_pca)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        return model, y_pred, r2, mse

    # Example usage:
    # ols_model, y_pred_ols, r2_ols, mse_ols = pcr(X_train, y_train, X_test, y_test, n_components=5)
    
    return pcr