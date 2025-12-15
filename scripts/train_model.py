"""
Script CLI para treinar qualquer modelo.

Uso:
    python scripts/train_model.py --model xgboost --output results/xgb_baseline.pkl
"""
import argparse
import sys
sys.path.insert(0, '.')

import numpy as np
from models.gradient_boosting import GradientBoostingModel, XGBoostModel
from models.kan import KANModel
import json


def load_data(data_dir='data/processed'):
    """Carrega dados processados."""
    X_train = np.load(f'{data_dir}/X_train.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    X_val = np.load(f'{data_dir}/X_val.npy')
    y_val = np.load(f'{data_dir}/y_val.npy')
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(model_type, config=None, data_dir='data/processed'):
    """
    Treina modelo.
    
    Args:
        model_type: 'gb', 'xgboost', 'kan'
        config: Dict com hiperparâmetros (opcional)
        data_dir: Diretório dos dados processados
    
    Returns:
        model treinado
    """
    # Carregar dados
    print(f"📂 Carregando dados de {data_dir}...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir)
    print(f"✅ Dados carregados: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Criar modelo
    if model_type == 'gb':
        model = GradientBoostingModel()
        default_config = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 3
        }
    elif model_type == 'xgboost':
        model = XGBoostModel()
        default_config = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 3
        }
    elif model_type == 'kan':
        model = KANModel()
        default_config = {
            'width': [X_train.shape[1], 20, 10, 1],
            'grid': 3,
            'k': 3
        }
    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")
    
    # Usar config fornecido ou default
    config = config or default_config
    print(f"\n🔧 Configuração: {config}")
    
    # Build
    model.build(**config)
    
    # Train
    if model_type in ['gb', 'xgboost']:
        model.train(X_train, y_train, X_val, y_val)
    else:  # KAN
        model.train(X_train, y_train, X_val, y_val, steps=100, lr=0.001)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Treinar modelo')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gb', 'xgboost', 'kan'],
                       help='Tipo de modelo')
    parser.add_argument('--output', type=str, required=True,
                       help='Caminho para salvar modelo')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON com hiperparâmetros')
    
    args = parser.parse_args()
    
    # Carregar config se fornecido
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Treinar
    model = train_model(args.model, config)
    
    # Salvar
    model.save(args.output)
    print(f"\n✅ Modelo salvo em: {args.output}")


if __name__ == '__main__':
    main()