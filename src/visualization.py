"""
Visualizações padronizadas para o projeto.
Gráficos seguem estilo dos slides 16-20 do PDF.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def plot_ks_statistic(y_true, y_proba, output_path=None, title="Gráfico KS"):
    """
    Gera o gráfico KS idêntico aos slides 16-18 do PDF.
    
    Mostra as curvas cumulativas das duas classes e a linha vertical
    no ponto de máximo KS.
    
    Args:
        y_true: Labels verdadeiros (0 ou 1)
        y_proba: Probabilidades preditas (0-1)
        output_path: Caminho para salvar (opcional)
        title: Título do gráfico
    """
    from src.metrics import calculate_ks
    
    # Calcular KS
    ks_stat, ks_threshold, df = calculate_ks(y_true, y_proba)
    
    # Criar figura
    plt.figure(figsize=(10, 6))
    
    # Plotar CDFs
    plt.plot(df.index / len(df), df['cumsum_class0'], 
             label='Classe 0 (Não Churn)', linewidth=2.5, color='#2E86AB')
    plt.plot(df.index / len(df), df['cumsum_class1'], 
             label='Classe 1 (Churn)', linewidth=2.5, color='#A23B72')
    
    # Linha vertical no KS máximo
    ks_idx = df['ks'].idxmax()
    ks_x = ks_idx / len(df)
    ks_y0 = df.loc[ks_idx, 'cumsum_class0']
    ks_y1 = df.loc[ks_idx, 'cumsum_class1']
    
    plt.vlines(ks_x, min(ks_y0, ks_y1), max(ks_y0, ks_y1),
               colors='black', linestyles='dashed', linewidth=2,
               label=f'KS = {ks_stat:.3f}')
    
    # Formatação
    plt.xlabel('Proporção da População', fontsize=12, fontweight='bold')
    plt.ylabel('Proporção Acumulada', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Salvar ou mostrar
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Gráfico KS salvo em: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return ks_stat


def plot_roc_curve(y_true, y_proba, output_path=None, title="Curva ROC"):
    """
    Plota curva ROC (slides 19-20 do PDF).
    
    Args:
        y_true: Labels verdadeiros
        y_proba: Probabilidades preditas
        output_path: Caminho para salvar (opcional)
        title: Título do gráfico
    """
    from sklearn.metrics import roc_auc_score
    
    # Calcular ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auroc = roc_auc_score(y_true, y_proba)
    
    # Plotar
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2.5, color='#2E86AB',
             label=f'AUROC = {auroc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, 
             label='Random Classifier', alpha=0.5)
    
    # Formatação
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Curva ROC salva em: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path=None, 
                         title="Matriz de Confusão", normalize=True):
    """
    Plota matriz de confusão (slide 15 do PDF).
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos (0 ou 1, não probabilidades)
        output_path: Caminho para salvar (opcional)
        title: Título do gráfico
        normalize: Se True, normaliza por linha (%)
    """
    # Calcular matriz
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2%'
    else:
        cm_display = cm
        fmt = 'd'
    
    # Plotar
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=['Não Churn', 'Churn'],
                yticklabels=['Não Churn', 'Churn'],
                cbar_kws={'label': 'Proporção' if normalize else 'Contagem'})
    
    plt.ylabel('Classe Real', fontsize=12, fontweight='bold')
    plt.xlabel('Classe Predita', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Matriz de Confusão salva em: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(y_true, y_proba, output_path=None,
                                title="Precision-Recall Curve"):
    """
    Plota curva Precision-Recall (slides 21-22 do PDF).
    
    Args:
        y_true: Labels verdadeiros
        y_proba: Probabilidades preditas
        output_path: Caminho para salvar (opcional)
        title: Título do gráfico
    """
    # Calcular curva
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Plotar
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2.5, color='#A23B72')
    
    # Formatação
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Precision-Recall salva em: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_comparison(results_dict, metric='ks', output_path=None):
    """
    Plota comparação entre modelos.
    
    Args:
        results_dict: Dict com {modelo: {train: metrics, val: metrics, test: metrics}}
        metric: Métrica para comparar ('ks', 'auroc', 'f1')
        output_path: Caminho para salvar (opcional)
    """
    models = list(results_dict.keys())
    train_scores = [results_dict[m]['train'][metric] for m in models]
    val_scores = [results_dict[m]['val'][metric] for m in models]
    test_scores = [results_dict[m]['test'][metric] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, train_scores, width, label='Train', color='#2E86AB')
    ax.bar(x, val_scores, width, label='Validation', color='#F18F01')
    ax.bar(x + width, test_scores, width, label='Test', color='#A23B72')
    
    ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
    ax.set_title(f'Comparação de Modelos - {metric.upper()}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Comparação salva em: {output_path}")
    else:
        plt.show()
    
    plt.close()