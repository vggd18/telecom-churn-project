import os
from datetime import datetime
import csv

def ensure_directories(base_dir='results'):
    """
    Cria a estrutura de pastas padrão se não existir.
    """
    subdirs = ['models', 'metrics', 'figures']
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    os.makedirs(base_dir, exist_ok=True)

def get_timestamped_path(filepath, subfolder=None):
    """
    Gera caminho com timestamp e organiza na subpasta correta.
    
    Args:
        filepath: Nome base ou caminho (ex: 'xgb_baseline.pkl')
        subfolder: 'models', 'metrics' ou 'figures' (opcional)
    
    Returns:
        Caminho completo (ex: 'results/models/xgb_baseline_20251216_103000.pkl')
    """
    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    
    # Timestamp atual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{name}_{timestamp}{ext}"
    
    if subfolder:
        if 'results' not in directory:
            directory = os.path.join('results', subfolder)
        else:
            base = directory.split('results')[0]
            directory = os.path.join(base, 'results', subfolder)
            
    os.makedirs(directory, exist_ok=True)
    
    return os.path.join(directory, new_filename)

def log_experiment(log_file='results/experiments_log.csv', model_name='', metrics=None, params=None):
    """
    Registra o experimento no CSV acumulativo (Slide 10 - Registrar desempenho).
    """
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['Timestamp', 'Model', 'KS_Test', 'AUROC_Test', 'Params'])
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ks = metrics.get('ks', 0) if metrics else 0
        auroc = metrics.get('auroc', 0) if metrics else 0
        
        writer.writerow([timestamp, model_name, f"{ks:.4f}", f"{auroc:.4f}", str(params)])