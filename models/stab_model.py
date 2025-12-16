import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel


class STabModel(BaseModel, nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,              # dim
        depth=2,                 # depth
        n_heads=4,               # heads
        ff_mult=4,               # U
        attn_dropout=0.1,        # attn_dropout
        ff_dropout=0.1,          # ff_dropout
        lr=1e-3,                 # lr
        weight_decay=0.0,        # weight_decay
        pooling="mean",          # cases
        name="STabTransformer",
        random_state=42
    ):
        BaseModel.__init__(self, name=name, random_state=random_state)
        nn.Module.__init__(self)

        self.input_dim = int(input_dim)
        self.pooling = pooling

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cada feature vira um token
        self.feature_embed = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=ff_dropout,
            activation="relu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_model, 1)
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.history = {"train_loss": [], "val_loss": []}
        self.model = None  # importante para evitar recursão

        self.to(self.device)

    def build(self, input_dim=None, **kwargs):
        if input_dim is not None:
            self.input_dim = int(input_dim)
        return self

    def forward(self, x):
        x = x.float()
        if x.dim() != 2:
            x = x.view(x.size(0), -1)

        x = x.unsqueeze(-1)           # [B, F, 1]
        x = self.feature_embed(x)     # [B, F, D]
        x = self.transformer(x)       # [B, F, D]

        if self.pooling == "mean":
            x = x.mean(dim=1)
        elif self.pooling == "cls":
            x = x[:, 0, :]
        else:
            raise ValueError("pooling must be 'mean' or 'cls'")

        logits = self.classifier(x).squeeze(-1)
        return logits

    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        epochs = int(kwargs.get("epochs", 20))
        batch_size = int(kwargs.get("batch_size", 256))

        X_train_t = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)

        X_val_t = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.float32).to(self.device)

        loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size,
            shuffle=True
        )

        for _ in range(epochs):
            nn.Module.train(self, True)
            total, n = 0.0, 0

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.forward(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

                total += loss.item() * xb.size(0)
                n += xb.size(0)

            train_loss = total / max(1, n)

            nn.Module.train(self, False)
            with torch.no_grad():
                val_logits = self.forward(X_val_t)
                val_loss = self.criterion(val_logits, y_val_t).item()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

        return self.history

    def predict_proba(self, X):
        nn.Module.train(self, False)
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.forward(X_t)
            return torch.sigmoid(logits).cpu().numpy()
