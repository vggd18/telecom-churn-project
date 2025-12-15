"""
Script de preprocessamento seguindo EXATAMENTE o PDF do professor.

Estratégia (Slide 33):
1. Separar classes
2. Split 50/25/25
3. Balancear TREINO E VALIDAÇÃO (oversampling)
4. Manter TESTE desbalanceado
5. Normalizar (fit no treino)
"""
import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
import json
import pickle


def run_data_pipeline(input_path, output_dir, random_state=42):
    """
    Pipeline completo seguindo diagrama do Slide 33.
    """
    print("="*60)
    print("🚀 PIPELINE DE PREPROCESSAMENTO (seguindo PDF)")
    print("="*60)
    
    # ==========================================
    # 1. Carregamento e Limpeza Inicial
    # ==========================================
    print("\n📂 1. Carregando dados...")
    df = pd.read_csv(input_path)
    print(f"   Shape original: {df.shape}")
    
    # Remover customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("   ✓ customerID removido")
    
    # Verificar missing
    missing = df.isnull().sum().sum()
    print(f"   Missing values: {missing}")
    if missing > 0:
        print(f"   ⚠️  Removendo linhas com missing")
        df = df.dropna()
        print(f"   Shape após limpeza: {df.shape}")
    
    # Converter Churn para binário
    if df["Churn"].dtype == 'object':
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0})
        print("   ✓ Churn convertido para 0/1")
    
    print(f"   Distribuição Churn: {df['Churn'].value_counts().to_dict()}")

    # ==========================================
    # 2. Particionamento (50/25/25) - Slide 31
    # ==========================================
    print("\n✂️ 2. Particionamento (50/25/25)...")
    
    # Separar por classe (Slide 31)
    class_0 = df[df["Churn"] == 0].copy()
    class_1 = df[df["Churn"] == 1].copy()
    
    print(f"   Classe 0 (No Churn): {len(class_0)}")
    print(f"   Classe 1 (Churn): {len(class_1)}")
    
    # Embaralhar
    class_0 = class_0.sample(frac=1, random_state=random_state).reset_index(drop=True)
    class_1 = class_1.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calcular tamanhos dos splits
    n_total_0 = len(class_0)
    n_train_0 = int(n_total_0 * 0.50)
    n_val_0 = int(n_total_0 * 0.25)
    
    n_total_1 = len(class_1)
    n_train_1 = int(n_total_1 * 0.50)
    n_val_1 = int(n_total_1 * 0.25)
    
    # Split classe 0
    c0_train = class_0.iloc[:n_train_0]
    c0_val = class_0.iloc[n_train_0 : n_train_0 + n_val_0]
    c0_test = class_0.iloc[n_train_0 + n_val_0:]
    
    # Split classe 1 (SEM OVERLAP)
    c1_train = class_1.iloc[:n_train_1]
    c1_val = class_1.iloc[n_train_1 : n_train_1 + n_val_1]
    c1_test = class_1.iloc[n_train_1 + n_val_1:]
    
    print(f"\n   Split Classe 0:")
    print(f"      Train: {len(c0_train)}")
    print(f"      Val: {len(c0_val)}")
    print(f"      Test: {len(c0_test)}")
    
    print(f"\n   Split Classe 1:")
    print(f"      Train: {len(c1_train)}")
    print(f"      Val: {len(c1_val)}")
    print(f"      Test: {len(c1_test)}")
    
    # Verificar ausência de overlap
    assert len(set(c1_train.index) & set(c1_val.index)) == 0, "❌ OVERLAP train-val!"
    assert len(set(c1_train.index) & set(c1_test.index)) == 0, "❌ OVERLAP train-test!"
    assert len(set(c1_val.index) & set(c1_test.index)) == 0, "❌ OVERLAP val-test!"
    print("   ✅ Verificação: Sem overlap entre splits")
    
    # ==========================================
    # 3. Concatenar (ANTES do balanceamento)
    # ==========================================
    df_train = pd.concat([c0_train, c1_train]).sample(frac=1, random_state=random_state)
    df_val = pd.concat([c0_val, c1_val]).sample(frac=1, random_state=random_state)
    df_test = pd.concat([c0_test, c1_test]).sample(frac=1, random_state=random_state)
    
    print(f"\n   Conjuntos (ANTES do balanceamento):")
    print(f"      Train: {df_train.shape} - Dist: {df_train['Churn'].value_counts().to_dict()}")
    print(f"      Val: {df_val.shape} - Dist: {df_val['Churn'].value_counts().to_dict()}")
    print(f"      Test: {df_test.shape} - Dist: {df_test['Churn'].value_counts().to_dict()}")
    
    # Separar X, y
    X_train = df_train.drop("Churn", axis=1)
    y_train = df_train["Churn"].values
    
    X_val = df_val.drop("Churn", axis=1)
    y_val = df_val["Churn"].values
    
    X_test = df_test.drop("Churn", axis=1)
    y_test = df_test["Churn"].values

    # ==========================================
    # 4. Pré-processamento (FIT APENAS NO TREINO)
    # ==========================================
    print("\n📏 4. Pré-processamento (encoding + normalização)...")
    
    # Identificar colunas
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()
    
    print(f"   Numéricas: {len(num_cols)} colunas")
    print(f"   Categóricas: {len(cat_cols)} colunas")

    # Criar preprocessor
    transformers = []
    
    if num_cols:
        transformers.append(
            ("num", StandardScaler(), num_cols)
        )
    
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", 
                                 sparse_output=False), cat_cols)
        )
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=False
    )
    
    # FIT + TRANSFORM no treino
    print("   Aplicando transformações...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # APENAS TRANSFORM em val e test
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"   ✓ Train processado: {X_train_processed.shape}")
    print(f"   ✓ Val processado: {X_val_processed.shape}")
    print(f"   ✓ Test processado: {X_test_processed.shape}")
    
    # Verificar normalização
    print(f"\n   Validação da normalização:")
    print(f"      Train mean: {X_train_processed.mean():.4f} (deve ser ~0)")
    print(f"      Train std: {X_train_processed.std():.4f} (deve ser ~1)")
    print(f"      Val mean: {X_val_processed.mean():.4f} (NÃO deve ser 0)")
    print(f"      Test mean: {X_test_processed.mean():.4f} (NÃO deve ser 0)")

    # ==========================================
    # 5. Balanceamento (TREINO E VALIDAÇÃO) - Slide 33
    # ==========================================
    print("\n⚖️ 5. Balanceamento (seguindo Slide 33)...")
    
    # TREINO: "Classe 2 (com repetição)"
    print(f"\n   TREINO (antes): {X_train_processed.shape}")
    print(f"      Distribuição: {np.bincount(y_train)}")
    
    ros_train = RandomOverSampler(random_state=random_state)
    X_train_balanced, y_train_balanced = ros_train.fit_resample(
        X_train_processed, y_train
    )
    
    print(f"   TREINO (depois): {X_train_balanced.shape}")
    print(f"      Distribuição: {np.bincount(y_train_balanced)}")
    
    # VALIDAÇÃO: "Classe 2 (25%) (com repetição)"
    print(f"\n   VALIDAÇÃO (antes): {X_val_processed.shape}")
    print(f"      Distribuição: {np.bincount(y_val)}")
    
    ros_val = RandomOverSampler(random_state=random_state)
    X_val_balanced, y_val_balanced = ros_val.fit_resample(
        X_val_processed, y_val
    )
    
    print(f"   VALIDAÇÃO (depois): {X_val_balanced.shape}")
    print(f"      Distribuição: {np.bincount(y_val_balanced)}")
    
    # TESTE: "Classe 2 (25%)" SEM "(com repetição)"
    print(f"\n   TESTE: {X_test_processed.shape}")
    print(f"      Distribuição: {np.bincount(y_test)} (mantida desbalanceada)")
    
    # ==========================================
    # 6. Salvamento dos Artefatos
    # ==========================================
    print("\n💾 6. Salvando artefatos...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar arrays (treino e validação BALANCEADOS, teste DESBALANCEADO)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train_balanced)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train_balanced)
    
    np.save(os.path.join(output_dir, "X_val.npy"), X_val_balanced)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val_balanced)
    
    np.save(os.path.join(output_dir, "X_test.npy"), X_test_processed)  # DESBALANCEADO
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    # Salvar preprocessor
    with open(os.path.join(output_dir, "preprocessor.pkl"), 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Salvar metadata
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except:
        feature_names = None
    
    metadata = {
        'train_shape': list(X_train_balanced.shape),
        'val_shape': list(X_val_balanced.shape),
        'test_shape': list(X_test_processed.shape),
        'n_features': int(X_train_balanced.shape[1]),
        'feature_names': feature_names,
        'train_class_dist': {
            'class_0': int(np.sum(y_train_balanced == 0)),
            'class_1': int(np.sum(y_train_balanced == 1)),
            'balanced': True
        },
        'val_class_dist': {
            'class_0': int(np.sum(y_val_balanced == 0)),
            'class_1': int(np.sum(y_val_balanced == 1)),
            'balanced': True
        },
        'test_class_dist': {
            'class_0': int(np.sum(y_test == 0)),
            'class_1': int(np.sum(y_test == 1)),
            'balanced': False  # TESTE DESBALANCEADO
        },
        'random_state': random_state,
        'strategy': 'PDF_Slide_33: train+val balanced, test imbalanced'
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Salvos em: {output_dir}")
    print(f"      - X_train.npy: {X_train_balanced.shape} (BALANCEADO)")
    print(f"      - y_train.npy: {y_train_balanced.shape}")
    print(f"      - X_val.npy: {X_val_balanced.shape} (BALANCEADO)")
    print(f"      - y_val.npy: {y_val_balanced.shape}")
    print(f"      - X_test.npy: {X_test_processed.shape} (DESBALANCEADO)")
    print(f"      - y_test.npy: {y_test.shape}")
    print(f"      - preprocessor.pkl")
    print(f"      - metadata.json")
    
    # ==========================================
    # 7. Validação Final
    # ==========================================
    print("\n" + "="*60)
    print("🔍 VALIDAÇÃO FINAL")
    print("="*60)
    
    print(f"\n✅ Shapes:")
    print(f"   Train: {X_train_balanced.shape} (balanceado)")
    print(f"   Val: {X_val_balanced.shape} (balanceado)")
    print(f"   Test: {X_test_processed.shape} (desbalanceado)")
    
    print(f"\n✅ Distribuições:")
    print(f"   Train: {dict(zip(*np.unique(y_train_balanced, return_counts=True)))} ← balanceado")
    print(f"   Val: {dict(zip(*np.unique(y_val_balanced, return_counts=True)))} ← balanceado")
    print(f"   Test: {dict(zip(*np.unique(y_test, return_counts=True)))} ← desbalanceado (real)")
    
    print(f"\n✅ Normalização:")
    print(f"   Train: mean={X_train_balanced.mean():.4f}, std={X_train_balanced.std():.4f}")
    
    print(f"\n✅ Dados limpos:")
    print(f"   NaN no train: {np.isnan(X_train_balanced).sum()}")
    print(f"   NaN no val: {np.isnan(X_val_balanced).sum()}")
    print(f"   NaN no test: {np.isnan(X_test_processed).sum()}")
    
    print("\n" + "="*60)
    print("🎉 PIPELINE CONCLUÍDO (Slide 33 do PDF)")
    print("="*60)


if __name__ == "__main__":
    run_data_pipeline(
        input_path="data/raw/telecom_churn.csv",
        output_dir="data/processed/"
    )