"""
Script CLI para avaliar modelo salvo.
Uso: python scripts/evaluate_model.py --model results/models/xgb_baseline.pkl --type xgboost --output metrics_xgb.json
"""
import argparse
import sys
import os
import json
import numpy as np

# Adicionar diretório raiz ao path
sys.path.insert(0, '.')

from src.utils import get_timestamped_path, ensure_directories, log_experiment
from models.gradient_boosting import GradientBoostingModel, XGBoostModel
from models.kan import KANModel
from src.metrics import calculate_all_metrics
from src.visualization import plot_ks_statistic, plot_roc_curve, plot_confusion_matrix

def load_data(data_dir='data/processed'):
    print(f"📂 Carregando dados de {data_dir}...")
    try:
        X_train = np.load(f'{data_dir}/X_train.npy')
        y_train = np.load(f'{data_dir}/y_train.npy')
        X_val = np.load(f'{data_dir}/X_val.npy')
        y_val = np.load(f'{data_dir}/y_val.npy')
        X_test = np.load(f'{data_dir}/X_test.npy')
        y_test = np.load(f'{data_dir}/y_test.npy')
        return X_train, y_train, X_val, y_val, X_test, y_test
    except FileNotFoundError as e:
        print(f"❌ Erro ao carregar dados: {e}")
        sys.exit(1)

def evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "="*60)
    print(f"📊 AVALIAÇÃO: {model.name}")
    print("="*60)
    
    y_train_pred = model.predict_proba(X_train)
    y_val_pred = model.predict_proba(X_val)
    y_test_pred = model.predict_proba(X_test)
    
    metrics_train = calculate_all_metrics(y_train, y_train_pred)
    metrics_val = calculate_all_metrics(y_val, y_val_pred)
    metrics_test = calculate_all_metrics(y_test, y_test_pred)
    
    print(f"\n{'Dataset':<10} {'KS':<8} {'AUROC':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print("-"*65)
    print(f"{'Train':<10} {metrics_train['ks']:<8.4f} {metrics_train['auroc']:<8.4f} {metrics_train['precision']:<10.4f} {metrics_train['recall']:<8.4f} {metrics_train['f1']:<8.4f}")
    print(f"{'Val':<10} {metrics_val['ks']:<8.4f} {metrics_val['auroc']:<8.4f} {metrics_val['precision']:<10.4f} {metrics_val['recall']:<8.4f} {metrics_val['f1']:<8.4f}")
    print(f"{'Test':<10} {metrics_test['ks']:<8.4f} {metrics_test['auroc']:<8.4f} {metrics_test['precision']:<10.4f} {metrics_test['recall']:<8.4f} {metrics_test['f1']:<8.4f}")
    
    return {'train': metrics_train, 'val': metrics_val, 'test': metrics_test}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--type', type=str, required=True, choices=['gb', 'xgboost', 'kan'])
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    if args.type == 'gb': model = GradientBoostingModel()
    elif args.type == 'xgboost': model = XGBoostModel()
    else: model = KANModel()
    
    model.load(args.model)
    print(f"✅ Modelo carregado: {args.model}")
    
    metrics = evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test)
    
    if args.output:
        print("\n💾 Salvando resultados...")
        ensure_directories()
        
        # Salvar JSON
        metrics_path = get_timestamped_path(args.output, subfolder='metrics')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"   ✅ JSON salvo em: {metrics_path}")
        
        # Log
        log_experiment(model_name=f"{model.name} ({args.type})", metrics=metrics['test'], params=model.get_config())
        
        # Gráficos
        print("   📊 Gerando gráficos...")
        y_test_pred = model.predict_proba(X_test)
        base_filename = os.path.basename(metrics_path).replace('.json', '')
        
        plot_ks_statistic(y_test, y_test_pred, output_path=os.path.join('results', 'figures', f"{base_filename}_ks.png"))
        plot_roc_curve(y_test, y_test_pred, output_path=os.path.join('results', 'figures', f"{base_filename}_roc.png"))
        
        y_test_class = (y_test_pred >= 0.5).astype(int)
        plot_confusion_matrix(y_test, y_test_class, output_path=os.path.join('results', 'figures', f"{base_filename}_cm.png"))
        
        print(f"   ✅ Gráficos salvos em results/figures/")

if __name__ == '__main__':
    main()