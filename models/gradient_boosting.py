"""
Gradient Boosting e XGBoost com interface padronizada.
"""
from models.base_model import BaseModel
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import pickle
import time

class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting (scikit-learn).
    """
    
    def __init__(self, name="GradientBoosting", random_state=42):
        super().__init__(name, random_state)
        self.config = {}
        self.training_time = 0
        
    def build(self, learning_rate=0.1, n_estimators=100, 
              max_depth=3, subsample=1.0, min_samples_leaf=1,
              **kwargs):
        """
        Constrói Gradient Boosting.
        
        Args:
            learning_rate: Taxa de aprendizado (0.01-0.1)
            n_estimators: Número de árvores (100-300)
            max_depth: Profundidade máxima (3-7)
            subsample: Fração de samples por árvore (0.8-1.0)
            min_samples_leaf: Min amostras por folha (1-5)
        """
        self.config = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'subsample': subsample,
            'min_samples_leaf': min_samples_leaf
        }
        
        self.model = GradientBoostingClassifier(
            loss='log_loss',  # 'deviance' deprecated
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
            criterion='friedman_mse',
            random_state=self.random_state,
            verbose=0,
            **kwargs
        )
        
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Treina o modelo.
        
        Note: Gradient Boosting do sklearn não usa validation set
        durante treino (diferente de XGBoost).
        """
        print(f"🚀 Treinando {self.name}...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        print(f"✅ Treinamento concluído em {self.training_time:.2f}s")
        
        return self
    
    def predict_proba(self, X):
        """Retorna probabilidades da classe positiva."""
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self):
        """Retorna importância das features."""
        return self.model.feature_importances_
    
    def save(self, filepath):
        """Salva modelo."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath):
        """Carrega modelo."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        return self
    
    def get_config(self):
        """Retorna configuração completa."""
        return {
            **super().get_config(),
            **self.config,
            'training_time': self.training_time
        }


class XGBoostModel(BaseModel):
    """
    XGBoost com early stopping.
    """
    
    def __init__(self, name="XGBoost", random_state=42):
        super().__init__(name, random_state)
        self.config = {}
        self.training_time = 0
        self.best_iteration = 0
        
    def build(self, learning_rate=0.1, n_estimators=100, 
              max_depth=3, subsample=1.0, colsample_bytree=1.0,
              gamma=0, reg_alpha=0, reg_lambda=1,
              **kwargs):
        """
        Constrói XGBoost.
        
        Args:
            learning_rate: Taxa de aprendizado (0.01-0.1)
            n_estimators: Número de árvores (100-300)
            max_depth: Profundidade máxima (3-7)
            subsample: Fração de samples por árvore (0.8-1.0)
            colsample_bytree: Fração de features por árvore (0.8-1.0)
            gamma: Min redução de loss para split (0-5)
            reg_alpha: Regularização L1 (0-1)
            reg_lambda: Regularização L2 (0-10)
        """
        self.config = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda
        }
        
        self.model = xgb.XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            **kwargs
        )
        
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              early_stopping_rounds=20, verbose=False):
        """
        Treina com early stopping (Compatível com XGBoost 2.0+).
        """
        print(f"🚀 Treinando {self.name}...")
        start_time = time.time()
        
        # Preparar eval_set
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        # FIX PARA XGBOOST 2.0+:
        # Early stopping agora é parâmetro do modelo, não do fit.
        # Usamos set_params para injetar dinamicamente.
        if eval_set and early_stopping_rounds:
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
        
        # O fit fica limpo, sem callbacks explícitos
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        self.training_time = time.time() - start_time
        
        # Recuperar melhor iteração (se early stopping foi usado)
        if hasattr(self.model, 'best_iteration'):
            self.best_iteration = self.model.best_iteration
        else:
            self.best_iteration = self.config['n_estimators']
            
        print(f"✅ Treinamento concluído em {self.training_time:.2f}s")
        if eval_set:
            print(f"   Melhor iteração: {self.best_iteration}")
        
        return self 
    
    def predict_proba(self, X):
        """Retorna probabilidades da classe positiva."""
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self):
        """Retorna importância das features."""
        return self.model.feature_importances_
    
    def save(self, filepath):
        """Salva modelo."""
        self.model.save_model(filepath)
    
    def load(self, filepath):
        """Carrega modelo."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(filepath)
        return self
    
    def get_config(self):
        """Retorna configuração completa."""
        return {
            **super().get_config(),
            **self.config,
            'training_time': self.training_time,
            'best_iteration': self.best_iteration
        }