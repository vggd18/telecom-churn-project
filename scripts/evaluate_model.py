"""
Script CLI para avaliar modelo salvo.

Uso:
    python scripts/evaluate_model.py --model results/xgb_baseline.pkl --type xgboost --output results/metrics_xgb.json
"""
import argparse
import sys
import os
sys.path.insert(0, '.')

import numpy as np
import json
from src.utils import get_timestamped_path
from models.gradient_boosting import GradientBoostingModel, XGBoostModel
from models.kan import KANModel
from src.metrics import calculate_all_metrics
from src.visualization import plot_ks_statistic, plot_roc_curve, plot_confusion_matrix


def load_data(data_dir='data/processed'):
    """Carrega dados processados."""
    X_train = np.load(f'{data_dir}/X_train.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    X_val = np.load(f'{data_dir}/X_val.npy')
    y_val = np.load(f'{data_dir}/y_val.npy')
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Avalia modelo em todos os conjuntos.
    
    Returns:
        dict com métricas
    """
    print("\n" + "="*60)
    print(f"📊 AVALIAÇÃO: {model.name}")
    print("="*60)
    
    # Predições
    y_train_pred = model.predict_proba(X_train)
    y_val_pred = model.predict_proba(X_val)
    y_test_pred = model.predict_proba(X_test)
    
    # Métricas
    metrics_train = calculate_all_metrics(y_train, y_train_pred)
    metrics_val = calculate_all_metrics(y_val, y_val_pred)
    metrics_test = calculate_all_metrics(y_test, y_test_pred)
    
    # Imprimir Tabela
    print(f"\n{'Dataset':<10} {'KS':<8} {'AUROC':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print("-"*65)
    print(f"{'Train':<10} {metrics_train['ks']:<8.4f} {metrics_train['auroc']:<8.4f} "
          f"{metrics_train['precision']:<10.4f} {metrics_train['recall']:<8.4f} {metrics_train['f1']:<8.4f}")
    print(f"{'Val':<10} {metrics_val['ks']:<8.4f} {metrics_val['auroc']:<8.4f} "
          f"{metrics_val['precision']:<10.4f} {metrics_val['recall']:<8.4f} {metrics_val['f1']:<8.4f}")
    print(f"{'Test':<10} {metrics_test['ks']:<8.4f} {metrics_test['auroc']:<8.4f} "
          f"{metrics_test['precision']:<10.4f} {metrics_test['recall']:<8.4f} {metrics_test['f1']:<8.4f}")
    
    # Overfitting check
    ks_diff = metrics_train['ks'] - metrics_test['ks']
    if ks_diff > 0.1:
        print(f"\n⚠️  ALERTA: Possível overfitting (KS train - test = {ks_diff:.3f})")
    else:
        print(f"\n✅ Generalização OK (KS train - test = {ks_diff:.3f})")
    
    return {
        'train': metrics_train,
        'val': metrics_val,
        'test': metrics_test
    }


def main():
    parser = argparse.ArgumentParser(description='Avaliar modelo')
    parser.add_argument('--model', type=str, required=True,
                       help='Caminho do modelo salvo')
    parser.add_argument('--type', type=str, required=True,
                       choices=['gb', 'xgboost', 'kan'],
                       help='Tipo de modelo')
    parser.add_argument('--output', type=str, default=None,
                       help='Salvar métricas em JSON e gráficos')
    
    args = parser.parse_args()
    
    # Carregar dados
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Carregar modelo
    if args.type == 'gb':
        model = GradientBoostingModel()
    elif args.type == 'xgboost':
        model = XGBoostModel()
    else:
        model = KANModel()
    
    model.load(args.model)
    print(f"✅ Modelo carregado: {args.model}")
    
    # Avaliar
    metrics = evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Salvar se especificado
    if args.output:
        final_output_path = get_timestamped_path(args.output)
        
        with open(final_output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✅ Métricas salvas em: {final_output_path}")
        print("📊 Gerando gráficos...")
        
        y_test_pred = model.predict_proba(X_test)
        base_path = final_output_path.replace('.json', '')
        ks_path = f"{base_path}_ks.png"
        roc_path = f"{base_path}_roc.png"
        cm_path = f"{base_path}_cm.png"
        
        # Plotar KS
        plot_ks_statistic(y_test, y_test_pred, output_path=ks_path, 
                         title=f"KS Plot - {model.name}")
        
        # Plotar ROC
        plot_roc_curve(y_test, y_test_pred, output_path=roc_path,
                      title=f"ROC Curve - {model.name}")
        
        # Plotar Matriz de Confusão
        y_test_class = (y_test_pred >= 0.5).astype(int)
        plot_confusion_matrix(y_test, y_test_class, output_path=cm_path,
                             title=f"Confusion Matrix - {model.name}")
        
        print(f"📈 Gráficos salvos:")
        print(f"   - {ks_path}")
        print(f"   - {roc_path}")
        print(f"   - {cm_path}")


if __name__ == '__main__':
    main()