from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from models.linear_models.base_regressor import BaseRegressor

class NeuralNetworkModel(BaseRegressor):
    """Feed-Forward Neural Network (FFN) basierend auf MLPRegressor nach Lecture Slides."""
    def __init__(self, n_stocks=None,
                 hidden_layer_sizes=(20,20,20),
                 activation='logistic',
                 alpha=0.0001,
                 learning_rate_init=0.001,
                 max_iter=50000,
                 early_stopping=True,
                 validation_fraction=0.2,
                 batch_size='auto'):
        super().__init__(n_stocks=n_stocks)
        self.hidden_layer_sizes  = hidden_layer_sizes
        self.activation          = activation
        self.alpha               = alpha
        self.learning_rate_init  = learning_rate_init
        self.max_iter            = max_iter
        self.early_stopping      = early_stopping
        self.validation_fraction = validation_fraction
        self.batch_size          = batch_size

    def build_pipeline(self):
        # Pipeline: Skalierung + MLPRegressor
        return Pipeline([
            ('scaler', StandardScaler()),
            ('nn',     MLPRegressor(
                            hidden_layer_sizes=self.hidden_layer_sizes,
                            activation=self.activation,
                            solver='adam',
                            alpha=self.alpha,
                            learning_rate_init=self.learning_rate_init,
                            max_iter=self.max_iter,
                            early_stopping=self.early_stopping,
                            validation_fraction=self.validation_fraction,
                            batch_size=self.batch_size
                        )
            )
        ])

    # Für Neuronale Netze genügt BaseRegressor.train(), evaluate(), plot_diagnostics()
    # Optional: Methode zur Ausgabe der Architektur
    def print_architecture(self):
        print(f"NN Architecture: layers={self.hidden_layer_sizes}, activation={self.activation}")
        print(f"Alpha={self.alpha}, lr_init={self.learning_rate_init}")
        print()
