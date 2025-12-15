"""
Métricas de avaliação para classificação binária.
KS (Kolmogorov-Smirnov) como métrica principal.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    roc_curve
)


def calculate_ks(y_true, y_proba):
    """
    Calcula KS (Kolmogorov-Smirnov) para classificação binária.
    
    KS = max(|CDF_classe1 - CDF_classe0|)
    
    Esta é a métrica PRINCIPAL do projeto (slides 16-18 do PDF).
    
    Args:
        y_true: Labels verdadeiros (0 ou 1)
        y_proba: Probabilidades preditas da classe positiva (0-1)
    
    Returns:
        ks_stat: Valor do KS (0-1, quanto maior melhor)
        ks_threshold: Threshold onde KS é máximo
        df: DataFrame com curvas cumulativas (para plotar)
    """
    # Criar DataFrame e ordenar por probabilidade decrescente
    df = pd.DataFrame({
        'prob': y_proba,
        'label': y_true
    }).sort_values('prob', ascending=False).reset_index(drop=True)
    
    # Calcular proporções cumulativas de cada classe
    total_class_1 = (df['label'] == 1).sum()
    total_class_0 = (df['label'] == 0).sum()
    
    if total_class_1 == 0 or total_class_0 == 0:
        # Se uma classe está ausente, KS não faz sentido
        return 0.0, 0.5, df
    
    # CDF (Cumulative Distribution Function)
    df['cumsum_class1'] = (df['label'] == 1).cumsum() / total_class_1
    df['cumsum_class0'] = (df['label'] == 0).cumsum() / total_class_0
    
    # KS = máxima diferença entre as CDFs
    df['ks'] = np.abs(df['cumsum_class1'] - df['cumsum_class0'])
    
    ks_stat = df['ks'].max()
    ks_idx = df['ks'].idxmax()
    ks_threshold = df.loc[ks_idx, 'prob']
    
    return ks_stat, ks_threshold, df


def calculate_all_metrics(y_true, y_proba, threshold=0.5):
    """
    Calcula todas as métricas exigidas pelo projeto.
    
    Args:
        y_true: Labels verdadeiros (0 ou 1)
        y_proba: Probabilidades preditas (0-1)
        threshold: Threshold para conversão em classe (default: 0.5)
    
    Returns:
        dict com KS, AUROC, precision, recall, f1, confusion_matrix
    """
    # Converter probabilidades em predições binárias
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calcular KS (métrica principal)
    ks_stat, ks_threshold, _ = calculate_ks(y_true, y_proba)
    
    # AUROC
    try:
        auroc = roc_auc_score(y_true, y_proba)
    except:
        auroc = 0.5  # Se falhar (ex: só uma classe presente)
    
    # Métricas de classificação
    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
    except:
        precision = recall = f1 = 0.0
        cm = np.zeros((2, 2))
    
    return {
        'ks': float(ks_stat),
        'ks_threshold': float(ks_threshold),
        'auroc': float(auroc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }


def calculate_mse(y_true, y_proba):
    """
    Calcula MSE (Mean Squared Error) das probabilidades.
    
    O PDF menciona MSE como métrica secundária (slide 13).
    Para classificação, é o erro quadrático médio das probabilidades.
    
    Args:
        y_true: Labels verdadeiros (0 ou 1)
        y_proba: Probabilidades preditas (0-1)
    
    Returns:
        mse: Erro quadrático médio
    """
    return float(np.mean((y_true - y_proba) ** 2))


def calculate_cross_entropy(y_true, y_proba, epsilon=1e-15):
    """
    Calcula Cross-Entropy (Binary).
    
    O PDF menciona Cross-Entropy como métrica de treinamento (slide 13).
    
    Args:
        y_true: Labels verdadeiros (0 ou 1)
        y_proba: Probabilidades preditas (0-1)
        epsilon: Valor pequeno para evitar log(0)
    
    Returns:
        cross_entropy: Log loss
    """
    # Clipar para evitar log(0)
    y_proba = np.clip(y_proba, epsilon, 1 - epsilon)
    
    # Binary cross-entropy
    ce = -np.mean(
        y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba)
    )
    
    return float(ce)