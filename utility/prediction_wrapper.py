# prediction_wrapper.py

import pandas as pd

class PredictionWrapper:
    """
    Wrappt beliebige BaseRegressor-Instanzen und liefert
    auf predict() einen DataFrame mit einer Spalte pro Modell.
    """

    def __init__(self, models: dict):
        """
        models: Dict[str, BaseRegressor]
            Schlüssel = Modellname, Wert = schon gefittete Modell-Instanz
        """
        self.models = models

    def predict(self, X):
        """
        Ruft .predict(X) für jedes Modell auf und gibt
        einen pandas.DataFrame zurück, in dem jede Spalte
        die Vorhersagen eines Modells enthält.
        """
        results = {}
        for name, model in self.models.items():
            results[name] = model.predict(X)
        return pd.DataFrame(results)
