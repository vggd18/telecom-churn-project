"""
KAN (Kolmogorov-Arnold Networks).
Implementação experimental conforme Slide 102.
"""
from models.base_model import BaseModel
import numpy as np
import time

# Tentar importar KAN
try:
    from kan import KAN
    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False
    print("⚠️  pykan não instalado. Instale com: pip install pykan")


class KANModel(BaseModel):
    """
    Kolmogorov-Arnold Network.
    """
    
    def __init__(self, name="KAN", random_state=42):
        super().__init__(name, random_state)
        self.config = {}
        self.training_time = 0
        self.history = {'train_loss': [], 'val_loss': []}
        
        if not KAN_AVAILABLE:
            raise ImportError("pykan não está instalado. Instale: pip install pykan")
    
    def build(self, input_dim=None, width=[20, 10, 1], grid=3, k=3, 
              seed=None, **kwargs):
        """
        Constrói KAN.
        Slide 102: Investigar KAN (versão mais recente).
        """
        # Se input_dim for passado, garante que width[0] combine
        if input_dim is not None and width[0] != input_dim:
            width = [input_dim] + width[1:]
            
        self.config = {
            'width': width,
            'grid': grid,
            'k': k,
            'seed': seed or self.random_state
        }
        
        try:
            # device='cpu' é mais seguro para datasets pequenos/médios sem setup CUDA complexo
            self.model = KAN(
                width=width,
                grid=grid,
                k=k,
                seed=self.config['seed'],
                **kwargs
            )
        except Exception as e:
            print(f"❌ Erro ao criar KAN: {e}")
            raise
        
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              steps=100, lr=0.001, lamb=0.0, 
              log_interval=10, **kwargs):
        """
        Treina KAN.
        """
        print(f"🚀 Treinando {self.name}...")
        start_time = time.time()
        
        X_train_tensor = X_train.astype(np.float32)
        y_train_tensor = y_train.reshape(-1, 1).astype(np.float32)
        
        dataset = {
            'train_input': X_train_tensor,
            'train_label': y_train_tensor
        }
        
        if X_val is not None:
            dataset['test_input'] = X_val.astype(np.float32)
            dataset['test_label'] = y_val.reshape(-1, 1).astype(np.float32)
        
        try:
            self.model.train(
                dataset,
                opt='Adam',
                steps=steps,
                lr=lr,
                lamb=lamb,
                log=log_interval,
                **kwargs
            )
            
            # Tentar extrair histórico (depende da versão do pykan)
            if hasattr(self.model, 'train_loss'):
                self.history['train_loss'] = self.model.train_loss
            if hasattr(self.model, 'test_loss'):
                self.history['val_loss'] = self.model.test_loss
                
        except Exception as e:
            print(f"❌ Erro KAN train: {e}")
            raise
        
        self.training_time = time.time() - start_time
        print(f"✅ Treino KAN finalizado: {self.training_time:.2f}s")
        return self
    
    def predict_proba(self, X):
        X_tensor = X.astype(np.float32)
        try:
            # KAN retorna logits ou valor de regressão
            predictions = self.model(X_tensor).flatten().detach().numpy()
            # Sigmoid para garantir [0,1]
            predictions = 1 / (1 + np.exp(-predictions))
            return np.clip(predictions, 0, 1)
        except Exception:
            # Fallback se detach/numpy falhar (versões diferentes)
            predictions = self.model(X_tensor).flatten()
            predictions = 1 / (1 + np.exp(-predictions))
            return np.clip(predictions, 0, 1)

    def save(self, filepath):
        try:
            self.model.save(filepath)
        except:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
    
    def load(self, filepath):
        import pickle
        try:
            self.model = KAN.load(filepath)
        except:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
        return self