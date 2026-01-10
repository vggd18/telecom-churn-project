import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base_model import BaseModel


class STabModel(BaseModel, nn.Module):
    def __init__(self, config=None, verbose=False):
        # Inicializa BaseModel e nn.Module
        BaseModel.__init__(self, name="STabTransformer", random_state=42)
        nn.Module.__init__(self)

        self.config = config or {}
        self.verbose = bool(self.config.get("verbose", verbose))

        # Hiperparâmetros
        self.d_model = int(self.config.get("d_model", 64))
        self.depth = int(self.config.get("depth", 2))
        self.n_heads = int(self.config.get("n_heads", 4))
        self.ff_mult = int(self.config.get("ff_mult", 4))
        self.attn_dropout = float(self.config.get("attn_dropout", 0.1))
        self.ff_dropout = float(self.config.get("ff_dropout", 0.1))
        self.lr = float(self.config.get("lr", 1e-3))
        self.weight_decay = float(self.config.get("weight_decay", 0.0))
        self.pooling = self.config.get("pooling", "mean")

        self.batch_size = int(self.config.get("batch_size", 256))
        self.epochs = int(self.config.get("epochs", 20))
        self.max_fail = int(self.config.get("max_fail", 20))

        self.input_dim = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Componentes serão criados no build()
        self.feature_embed = None
        self.transformer = None
        self.classifier = None
        self.criterion = nn.BCEWithLogitsLoss()

        self.history = {"train_loss": [], "val_loss": []}

        if self.verbose:
            print(f"STabModel inicializado no device: {self.device}")

    def build(self, input_dim=None, **kwargs):
        if input_dim is not None:
            self.input_dim = int(input_dim)

        if self.input_dim is None:
            raise ValueError("input_dim deve ser fornecido para construir o modelo.")

        # Camada de Embedding
        self.feature_embed = nn.Linear(1, self.d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * self.ff_mult,
            dropout=self.ff_dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)

        # Classificador Final
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.ff_dropout),
            nn.Linear(self.d_model, 1),
        )

        self.optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.to(self.device)
        return self

    def forward(self, x):
        x = x.float()
        if x.dim() != 2:
            x = x.view(x.size(0), -1)

        # [Batch, Features] -> [Batch, Features, 1] -> [Batch, Features, D_model]
        x = x.unsqueeze(-1)
        x = self.feature_embed(x)
        x = self.transformer(x)

        # Pooling
        if self.pooling == "mean":
            x = x.mean(dim=1)
        elif self.pooling == "cls":
            # Assumindo que o primeiro token seria CLS se implementado,
            # aqui pegamos o primeiro feature como proxy ou média é melhor.
            x = x[:, 0, :]
        else:
            raise ValueError("pooling must be 'mean' or 'cls'")

        logits = self.classifier(x).squeeze(-1)
        return logits

    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        # Garante que o modelo está construído
        if self.feature_embed is None:
            self.build(input_dim=X_train.shape[1])

        # Preparação dos dados
        X_train_t = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)
        X_val_t = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        fails = 0

        print(f"🚀 Iniciando treino: {self.epochs} épocas...")

        for epoch in range(1, self.epochs + 1):
            nn.Module.train(self, True)
            total_loss = 0.0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                self.optimizer.zero_grad()
                logits = self.forward(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * xb.size(0)

            avg_train_loss = total_loss / len(dataset)

            # Validação
            nn.Module.train(self, False)
            with torch.no_grad():
                val_logits = self.forward(X_val_t)
                val_loss = self.criterion(val_logits, y_val_t).item()

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)

            if self.verbose and epoch % self.config.get("log_every", 5) == 0:
                print(
                    f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}"
                )

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                fails = 0
            else:
                fails += 1
                if fails >= self.max_fail:
                    print(f"⏹️ Early stopping na época {epoch}")
                    break

        return self.history

    def predict_proba(self, X):
        nn.Module.train(self, False)
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.forward(X_t)
            return torch.sigmoid(logits).cpu().numpy()

    def save(self, filepath):
        """Salva pesos e configurações em .pth"""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "input_dim": self.input_dim,
            "history": self.history,
        }
        torch.save(checkpoint, filepath)
        if self.verbose:
            print(f"💾 Modelo STab salvo em: {filepath}")

    def load(self, filepath):
        """Carrega do arquivo .pth"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.config = checkpoint.get("config", self.config)
        self.input_dim = checkpoint.get("input_dim", self.input_dim)
        self.history = checkpoint.get("history", {})

        if self.input_dim and self.feature_embed is None:
            self.build(self.input_dim)

        self.load_state_dict(checkpoint["model_state_dict"])
        self.to(self.device)
        print(f"♻️ Modelo STab carregado de: {filepath}")
