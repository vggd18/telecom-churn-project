"""
Classe abstrata para padronizar interface.
"""

from abc import ABC, abstractmethod
import joblib
import os


class BaseModel(ABC):
    """
    Interface comum para todos os modelos.
    """

    def __init__(self, name, random_state=42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.history = {}
        self.config = {}

    @abstractmethod
    def build(self, input_dim=None, **kwargs):
        """Constrói a arquitetura do modelo."""
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        """Treina o modelo."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Retorna probabilidades."""
        pass

    def save(self, filepath):
        """
        Salva modelo usando joblib (padrão para Sklearn/XGBoost).
        Modelos PyTorch/Keras devem sobrescrever este método.
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self, filepath)
            print(f"💾 Modelo salvo (via joblib): {filepath}")
        except Exception as e:
            print(f"⚠️ Erro ao salvar modelo via joblib: {e}")

    def load(self, filepath):
        """Carrega modelo usando joblib."""
        try:
            loaded = joblib.load(filepath)
            self.__dict__.update(loaded.__dict__)
            print(f"♻️ Modelo carregado: {filepath}")
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo: {e}")

    def get_config(self):
        """Retorna configuração do modelo (para logging)."""
        return {"name": self.name, "random_state": self.random_state, **self.config}
