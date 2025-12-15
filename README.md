# Telecom Churn Prediction - Neural Networks Project

Projeto da disciplina de Redes Neurais (UFPE) para previsão de churn em telecom.

## 📊 Dataset
- **Fonte:** [Kaggle - Telecom Churn](https://www.kaggle.com/datasets/kapturovalexander/customers-churned-in-telecom-services/data)
- **Registros:** ~7.000
- **Features:** 19 variáveis independentes
- **Target:** Churn (binário)

## 🏗️ Modelos Implementados
1. MLP (Multilayer Perceptron)
2. Gradient Boosting
3. XGBoost
4. TabPFN v2
5. STab
6. KAN (Kolmogorov-Arnold Networks)
7. TabKAN

## 🚀 Como Rodar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Preprocessar dados
```bash
python scripts/prepare_data.py
```

### 3. Executar experimentos
Abrir notebooks em `experiments/`:
- `03_mlp_experiments.ipynb` (Integrante A)
- `04_transformers_experiments.ipynb` (Integrante B)
- `05_boosting_kan_experiments.ipynb` (Integrante C)

### 4. Consolidar resultados
```bash
jupyter notebook experiments/06_final_comparison.ipynb
```

## 📈 Métricas Principais
- **KS (Kolmogorov-Smirnov):** Métrica principal
- AUROC
- Precision, Recall, F1
- Matriz de Confusão

## 👥 Equipe
- Jonathas Vinicius: MLP
- Vítor Dias: Transformers tabulares
- Douglas Gemir: Boosting & KAN

## 📝 Estrutura do Projeto
```
├── data/              # Dados raw e processados
├── src/               # Módulos reutilizáveis
├── models/            # Classes dos modelos (POO)
├── experiments/       # Notebooks de experimentação
├── results/           # Logs, métricas, figuras
└── scripts/           # Scripts executáveis
```