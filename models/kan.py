"""
KAN (Kolmogorov-Arnold Networks).
Implementação experimental.
"""
from models.base_model import BaseModel
import numpy as np
import time
import torch
import os

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
    Wrapper com salvamento robusto (Arquitetura + Pesos).
    """
    
    def __init__(self, name="KAN", random_state=42):
        super().__init__(name, random_state)
        self.config = {}
        self.training_time = 0
        self.history = {'train_loss': [], 'val_loss': []}
        self.model = None # Garante que existe, mesmo que vazio
        
        if not KAN_AVAILABLE:
            raise ImportError("pykan não está instalado. Instale: pip install pykan")
    
    def build(self, input_dim=None, width=[20, 10, 1], grid=3, k=3, 
              seed=None, **kwargs):
        """
        Constrói KAN.
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
        
        # Limpar kwargs de treino que não servem pro construtor
        train_params = ['steps', 'lr', 'lamb', 'log_interval', 'early_stopping_rounds']
        init_kwargs = kwargs.copy()
        for param in train_params:
            init_kwargs.pop(param, None)

        try:
            self.model = KAN(
                width=width,
                grid=grid,
                k=k,
                seed=int(self.config['seed']),
                **init_kwargs
            )
        except Exception as e:
            print(f"❌ Erro ao criar KAN: {e}")
            raise
        
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              steps=50, lr=0.01, lamb=0.0, 
              log_interval=10, **kwargs):
        """
        Treina KAN.
        """
        print(f"🚀 Treinando {self.name}...")
        start_time = time.time()
        
        # Converter NumPy para Torch Tensor
        X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
        y_train_tensor = torch.from_numpy(y_train.reshape(-1, 1).astype(np.float32))
        
        dataset = {
            'train_input': X_train_tensor,
            'train_label': y_train_tensor
        }
        
        if X_val is not None:
            dataset['test_input'] = torch.from_numpy(X_val.astype(np.float32))
            dataset['test_label'] = torch.from_numpy(y_val.reshape(-1, 1).astype(np.float32))
        else:
            dataset['test_input'] = X_train_tensor
            dataset['test_label'] = y_train_tensor
        
        try:
            results = self.model.fit(
                dataset,
                opt='Adam',
                steps=steps,
                lr=lr,
                lamb=lamb,
                log=log_interval,
                **kwargs
            )
            
            if isinstance(results, dict) and 'train_loss' in results:
                self.history['train_loss'] = results['train_loss']
                
        except Exception as e:
            print(f"❌ Erro KAN train: {e}")
            pass
        
        self.training_time = time.time() - start_time
        print(f"✅ Treino KAN finalizado: {self.training_time:.2f}s")
        return self
    
    def predict_proba(self, X):
        if self.model is None:
            print("⚠️ Modelo KAN não foi carregado corretamente (None).")
            return np.zeros(len(X))

        X_tensor = torch.from_numpy(X.astype(np.float32))
        try:
            predictions = self.model(X_tensor)
            
            if hasattr(predictions, 'detach'):
                predictions = predictions.detach().numpy().flatten()
            else:
                predictions = predictions.flatten()
                
            predictions = 1 / (1 + np.exp(-predictions))
            return np.clip(predictions, 0, 1)
            
        except Exception as e:
            print(f"⚠️ Erro na predição KAN: {e}")
            return np.zeros(len(X))

    def save(self, filepath):
        """
        Salva um Checkpoint completo: Configuração + Pesos.
        """
        if self.model is None:
            print("⚠️ Tentativa de salvar modelo KAN vazio.")
            return

        checkpoint = {
            'config': self.config,
            'state_dict': self.model.state_dict()
        }
        try:
            torch.save(checkpoint, filepath)
        except Exception as e:
            print(f"❌ Erro ao salvar KAN: {e}")
    
    def load(self, filepath):
        """
        Carrega Checkpoint, reconstrói a arquitetura e carrega os pesos.
        """
        if not os.path.exists(filepath):
            print(f"❌ Arquivo não encontrado: {filepath}")
            return self

        try:
            # Carrega o dicionário salvo
            checkpoint = torch.load(filepath)
            
            # 1. Recupera a configuração e RECONSTRÓI o modelo
            if 'config' in checkpoint:
                self.config = checkpoint['config']
                # Chama o build internamente para criar o esqueleto self.model
                self.build(**self.config)
            else:
                print("⚠️ Aviso: Arquivo antigo sem 'config'. Tentando carregar apenas pesos...")
                # Se for arquivo antigo, precisamos torcer para o build já ter sido chamado externamente
                # ou falhará.
            
            # 2. Carrega os pesos no esqueleto pronto
            if 'state_dict' in checkpoint and self.model is not None:
                self.model.load_state_dict(checkpoint['state_dict'])
            elif self.model is None:
                print("❌ Falha: Não foi possível reconstruir a arquitetura do modelo.")
                
        except Exception as e:
            print(f"❌ Erro crítico ao carregar KAN: {e}")
            
        return self