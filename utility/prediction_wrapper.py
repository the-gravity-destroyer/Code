# prediction_wrapper.py

import pandas as pd

class PredictionWrapper:
    """
    Wraps any BaseRegressor instance
    and returns a DataFrame with one column per model on predict().
    """

    def __init__(self, models: dict):
        """
        models: Dict[str, BaseRegressor]
        Key = model name, value = already fitted model instance
        """
        self.models = models

    def predict(self, X):
        """
        Calls .predict(X) for each model and returns
        a pandas.DataFrame where each column
        contains the predictions of a model.
        """
        results = {}
        for name, model in self.models.items():
            results[name] = model.predict(X)
        return pd.DataFrame(results)
