"""
Classe abstrata para padronizar interface.
"""
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Interface comum para todos os modelos.
    """
    
    def __init__(self, name, random_state=42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.history = {}
        
    @abstractmethod
    def build(self, input_dim=None, **kwargs):
        """
        Constrói a arquitetura do modelo.
        input_dim é opcional pois árvores (XGB/GB) inferem isso no fit.
        """
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
        """Salva modelo."""
        pass
    
    def load(self, filepath):
        """Carrega modelo."""
        pass
    
    def get_config(self):
        """Retorna configuração do modelo (para logging)."""
        return {
            'name': self.name,
            'random_state': self.random_state
        }