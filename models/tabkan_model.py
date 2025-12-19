import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base_model import BaseModel


class KANLayer(nn.Module):
    """
    KANLayer simples (piecewise linear em grid) por feature:
    - Para cada feature x_j, aproxima f_j(x_j) por interpolação linear em um grid
    - Soma as contribuições das features e produz out_dim

    coeff: [in_dim, grid_size, out_dim]
    """
    def __init__(self, in_dim, out_dim, grid_size=16, x_min=-3.0, x_max=3.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.grid_size = int(grid_size)

        grid = torch.linspace(x_min, x_max, steps=self.grid_size)
        self.register_buffer("grid", grid)  # [G]

        self.coeff = nn.Parameter(torch.randn(self.in_dim, self.grid_size, self.out_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, x):
        """
        x: [B, in_dim]
        retorna: [B, out_dim]
        """
        x = x.float()
        if x.dim() != 2:
            x = x.view(x.size(0), -1)

        B, F = x.shape
        if F != self.in_dim:
            raise ValueError(f"KANLayer esperava in_dim={self.in_dim}, veio {F}")

        g = self.grid
        G = g.numel()

        # idx em [1..G-1] para usar (idx-1, idx)
        idx = torch.bucketize(x, g).clamp(1, G - 1)  # [B, F]

        g0 = g[idx - 1]  # [B, F]
        g1 = g[idx]      # [B, F]
        denom = (g1 - g0).clamp_min(1e-8)
        t = (x - g0) / denom  # [B, F]

        w0 = (1.0 - t).unsqueeze(-1)  # [B, F, 1]
        w1 = t.unsqueeze(-1)          # [B, F, 1]

        # Seleciona coef por feature e pelos dois nós do grid
        feat_idx = torch.arange(self.in_dim, device=x.device)[None, :].expand(B, -1)  # [B, F]
        c0 = self.coeff[feat_idx, (idx - 1), :]  # [B, F, out_dim]
        c1 = self.coeff[feat_idx, idx, :]        # [B, F, out_dim]

        y = (w0 * c0 + w1 * c1).sum(dim=1) + self.bias  # [B, out_dim]
        return y


class TabKANModel(BaseModel, nn.Module):
    """
    TabKAN no mesmo padrão do STab:
    - __init__(config=None, verbose=False) com defaults via .get()
    - build(input_dim)
    - train(...) com early stopping por max_fail
    - predict_proba(...)
    """

    def __init__(self, config=None, verbose=False):
        BaseModel.__init__(self, name="TabKAN", random_state=42)
        nn.Module.__init__(self)

        self.config = config or {}
        self.verbose = bool(self.config.get("verbose", verbose))

        # ===== Hyperparams (defaults seguros) =====
        self.depth        = int(self.config.get("depth", 2))
        self.hidden_dim   = int(self.config.get("hidden_dim", 64))
        self.grid_size    = int(self.config.get("grid_size", 16))
        self.x_min        = float(self.config.get("x_min", -3.0))
        self.x_max        = float(self.config.get("x_max",  3.0))

        self.dropout      = float(self.config.get("dropout", 0.1))
        self.lr           = float(self.config.get("lr", 1e-3))
        self.weight_decay = float(self.config.get("weight_decay", 0.0))
        self.activation   = self.config.get("activation", "relu")

        self.batch_size   = int(self.config.get("batch_size", 256))
        self.epochs       = int(self.config.get("epochs", 30))
        self.max_fail     = int(self.config.get("max_fail", 10))

        # ===== runtime =====
        self.input_dim = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()

        self.history = {"train_loss": [], "val_loss": []}
        self.model = None  # manter padrão pra evitar recursão em frameworks

        self.to(self.device)

        if self.verbose:
            print("========== TabKAN INIT ==========")
            print("Device:", self.device)
            print("Config efetivo:")
            print(f" depth={self.depth} hidden_dim={self.hidden_dim} grid_size={self.grid_size} x_range=[{self.x_min},{self.x_max}]")
            print(f" activation={self.activation} dropout={self.dropout}")
            print(f" lr={self.lr} weight_decay={self.weight_decay}")
            print(f" batch_size={self.batch_size} epochs={self.epochs} max_fail={self.max_fail}")
            print("================================\n")

    def _get_activation(self):
        act = str(self.activation).lower()
        if act == "relu":
            return nn.ReLU()
        if act == "tanh":
            return nn.Tanh()
        if act == "sigmoid":
            return nn.Sigmoid()
        raise ValueError("activation must be: relu | tanh | sigmoid")

    def build(self, input_dim=None, **kwargs):
        if input_dim is not None:
            self.input_dim = int(input_dim)

        if self.input_dim is None:
            raise ValueError("TabKANModel.build precisa de input_dim")

        act = self._get_activation()

        layers = []
        in_dim = self.input_dim

        for _ in range(self.depth):
            layers.append(KANLayer(
                in_dim=in_dim,
                out_dim=self.hidden_dim,
                grid_size=self.grid_size,
                x_min=self.x_min,
                x_max=self.x_max
            ))
            layers.append(act)
            layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim

        layers.append(nn.Linear(self.hidden_dim, 1))
        self.net = nn.Sequential(*layers).to(self.device)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.verbose:
            print(f"[build] input_dim registrado = {self.input_dim}")
        return self

    def forward(self, x):
        x = x.float()
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        if self.net is None:
            raise RuntimeError("Você precisa chamar build(input_dim) antes de treinar/predizer.")
        return self.net(x).squeeze(-1)

    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        epochs = int(kwargs.get("epochs", self.epochs))
        batch_size = int(kwargs.get("batch_size", self.batch_size))
        max_fail = int(kwargs.get("max_fail", self.max_fail))

        X_train_t = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)

        X_val_t = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.float32).to(self.device)

        loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

        if self.verbose:
            print("========== TRAIN START ==========")
            print(f"epochs={epochs}, batch_size={batch_size}, max_fail={max_fail}")
            print(f"n_train={len(X_train_t)}, n_val={len(X_val_t)}, num_batches/epoch={len(loader)}")
            print("================================\n")

        best_val = float("inf")
        fails = 0

        for ep in range(1, epochs + 1):
            t0 = time.time()
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

            if self.verbose:
                print(f"[epoch {ep}/{epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={time.time()-t0:.1f}s")

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                fails = 0
            else:
                fails += 1
                if fails >= max_fail:
                    if self.verbose:
                        print(f"EARLY STOP: sem melhora por {max_fail} épocas.")
                        print("========== TRAIN END ==========\n")
                    break

        if self.verbose and fails < max_fail:
            print("========== TRAIN END ==========\n")

        return self.history

    def predict_proba(self, X):
        nn.Module.train(self, False)
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.forward(X_t)
            return torch.sigmoid(logits).cpu().numpy()
