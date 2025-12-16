import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, mean_squared_error

from base_model import BaseModel


class MLPModel(BaseModel):
    """
    Implementação de MLP usando scikit-learn.
    """

    def build(
        self,
        input_dim=None,
        hidden_layer_sizes=(10,),
        activation="relu",
        learning_rate_init=0.001,
        alpha=0.0001,
        max_iter=10000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        **kwargs,
    ):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=self.random_state,
            **kwargs,
        )

        self.config = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "learning_rate_init": learning_rate_init,
            "alpha": alpha,
            "max_iter": max_iter,
            "early_stopping": early_stopping,
            "validation_fraction": validation_fraction,
            "n_iter_no_change": n_iter_no_change,
        }

    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        self.model.fit(X_train, y_train)

        # Probabilidades
        train_proba = self.model.predict_proba(X_train)[:, 1]
        val_proba = self.model.predict_proba(X_val)[:, 1]

        # Métricas exigidas no projeto
        self.history = {
            "train_logloss": log_loss(y_train, train_proba),
            "val_logloss": log_loss(y_val, val_proba),
            "train_mse": mean_squared_error(y_train, train_proba),
            "val_mse": mean_squared_error(y_val, val_proba),
            "n_iter": self.model.n_iter_,
        }

        return self.history

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, filepath):
        joblib.dump(
            {
                "model": self.model,
                "config": self.config,
                "history": self.history,
            },
            filepath,
        )

    def load(self, filepath):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.config = data.get("config", {})
        self.history = data.get("history", {})
