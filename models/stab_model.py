import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base_model import BaseModel


class STabModel(BaseModel, nn.Module):
    def __init__(self, config=None, verbose=False):
        BaseModel.__init__(self, name="STabTransformer", random_state=42)
        nn.Module.__init__(self)

        self.config = config or {}
        self.verbose = bool(self.config.get("verbose", verbose))

        # Defaults seguros (nunca mais KeyError)
        self.d_model      = int(self.config.get("d_model", 64))
        self.depth        = int(self.config.get("depth", 2))
        self.n_heads      = int(self.config.get("n_heads", 4))
        self.ff_mult      = int(self.config.get("ff_mult", 4))
        self.attn_dropout = float(self.config.get("attn_dropout", 0.1))  # não separado no layer padrão
        self.ff_dropout   = float(self.config.get("ff_dropout", 0.1))
        self.lr           = float(self.config.get("lr", 1e-3))
        self.weight_decay = float(self.config.get("weight_decay", 0.0))
        self.pooling      = self.config.get("pooling", "mean")

        self.batch_size   = int(self.config.get("batch_size", 256))
        self.epochs       = int(self.config.get("epochs", 20))
        self.max_fail     = int(self.config.get("max_fail", 20))

        self.input_dim = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cada feature vira um token
        self.feature_embed = nn.Linear(1, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * self.ff_mult,
            dropout=self.ff_dropout,
            activation="relu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.ff_dropout),
            nn.Linear(self.d_model, 1)
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.history = {"train_loss": [], "val_loss": []}
        self.model = None

        self.to(self.device)

        if self.verbose:
            self._debug_init()

    def _debug(self, msg):
        if self.verbose:
            print(msg)

    def _debug_init(self):
        self._debug("========== STabModel INIT ==========")
        self._debug(f"Device: {self.device}")
        self._debug("Config efetivo:")
        self._debug(f"  d_model={self.d_model}, depth={self.depth}, n_heads={self.n_heads}, ff_mult={self.ff_mult}")
        self._debug(f"  ff_dropout={self.ff_dropout}, attn_dropout={self.attn_dropout} (nota: não separado no layer)")
        self._debug(f"  lr={self.lr}, weight_decay={self.weight_decay}, pooling={self.pooling}")
        self._debug(f"  batch_size={self.batch_size}, epochs={self.epochs}, max_fail={self.max_fail}")
        self._debug("===================================\n")

    def build(self, input_dim=None, **kwargs):
        if input_dim is not None:
            self.input_dim = int(input_dim)
            self._debug(f"[build] input_dim registrado = {self.input_dim}")
        return self

    def forward(self, x):
        x = x.float()
        if x.dim() != 2:
            x = x.view(x.size(0), -1)

        # [B, F] -> [B, F, 1] -> [B, F, D]
        x = x.unsqueeze(-1)
        x = self.feature_embed(x)
        x = self.transformer(x)

        if self.pooling == "mean":
            x = x.mean(dim=1)
        elif self.pooling == "cls":
            x = x[:, 0, :]
        else:
            raise ValueError("pooling must be 'mean' or 'cls'")

        logits = self.classifier(x).squeeze(-1)
        return logits

    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        epochs = int(kwargs.get("epochs", self.epochs))
        batch_size = int(kwargs.get("batch_size", self.batch_size))
        max_fail = int(kwargs.get("max_fail", self.max_fail))

        log_every = int(kwargs.get("log_every", self.config.get("log_every", 20)))  # batches
        debug_shapes = bool(kwargs.get("debug_shapes", self.config.get("debug_shapes", True)))
        debug_grads = bool(kwargs.get("debug_grads", self.config.get("debug_grads", False)))

        X_train_np = np.asarray(X_train)
        y_train_np = np.asarray(y_train)
        X_val_np = np.asarray(X_val)
        y_val_np = np.asarray(y_val)

        # Tensores
        X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_np, dtype=torch.float32)

        X_val_t = torch.tensor(X_val_np, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val_np, dtype=torch.float32).to(self.device)

        loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

        if self.verbose:
            self._debug("========== TRAIN START ==========")
            self._debug(f"epochs={epochs}, batch_size={batch_size}, max_fail={max_fail}")
            self._debug(f"n_train={len(X_train_t)}, n_val={len(X_val_t)}")
            self._debug(f"X_train dtype={X_train_t.dtype}, y_train dtype={y_train_t.dtype}")
            self._debug(f"X_val device={X_val_t.device}, y_val device={y_val_t.device}")
            self._debug(f"num_batches/epoch={len(loader)}")
            self._debug("================================\n")

        best_val = float("inf")
        fails = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            nn.Module.train(self, True)

            total_loss = 0.0
            n = 0

            # debug de shapes em 1º batch
            first_batch_logged = False

            for bidx, (xb, yb) in enumerate(loader, start=1):
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                if self.verbose and debug_shapes and not first_batch_logged:
                    self._debug(f"[epoch {epoch}] batch 1 shapes:")
                    self._debug(f"  xb: {tuple(xb.shape)}  yb: {tuple(yb.shape)}")
                    # shapes internas rápidas
                    with torch.no_grad():
                        tmp = xb.float()
                        tmp = tmp.unsqueeze(-1)
                        self._debug(f"  xb.unsqueeze(-1): {tuple(tmp.shape)}")
                        tmp = self.feature_embed(tmp)
                        self._debug(f"  feature_embed: {tuple(tmp.shape)}")
                        tmp = self.transformer(tmp)
                        self._debug(f"  transformer out: {tuple(tmp.shape)}")
                        if self.pooling == "mean":
                            tmp2 = tmp.mean(dim=1)
                        else:
                            tmp2 = tmp[:, 0, :]
                        self._debug(f"  pooling '{self.pooling}': {tuple(tmp2.shape)}")
                        tmp3 = self.classifier(tmp2).squeeze(-1)
                        self._debug(f"  logits: {tuple(tmp3.shape)}")
                    first_batch_logged = True

                self.optimizer.zero_grad(set_to_none=True)

                logits = self.forward(xb)
                loss = self.criterion(logits, yb)

                loss.backward()

                if self.verbose and debug_grads and (bidx % log_every == 0):
                    # norma de gradiente (rápido)
                    total_norm = 0.0
                    for p in self.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2).item()
                            total_norm += param_norm ** 2
                    total_norm = total_norm ** 0.5
                    self._debug(f"[epoch {epoch}][batch {bidx}] grad_norm(L2)={total_norm:.4f}")

                self.optimizer.step()

                total_loss += loss.item() * xb.size(0)
                n += xb.size(0)

                if self.verbose and (bidx % log_every == 0):
                    lr_now = self.optimizer.param_groups[0]["lr"]
                    self._debug(f"[epoch {epoch}][batch {bidx}/{len(loader)}] loss={loss.item():.6f} lr={lr_now:g}")

            train_loss = total_loss / max(1, n)

            # validação
            nn.Module.train(self, False)
            with torch.no_grad():
                val_logits = self.forward(X_val_t)
                val_loss = self.criterion(val_logits, y_val_t).item()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            dt = time.time() - t0
            lr_now = self.optimizer.param_groups[0]["lr"]

            # early stopping
            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                fails = 0
            else:
                fails += 1

            if self.verbose:
                self._debug(
                    f"[epoch {epoch}/{epochs}] "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                    f"best_val={best_val:.6f} fails={fails}/{max_fail} "
                    f"lr={lr_now:g} time={dt:.1f}s"
                )

            if fails >= max_fail:
                if self.verbose:
                    self._debug(f"EARLY STOP: sem melhora por {max_fail} épocas.")
                    self._debug("========== TRAIN END ==========\n")
                break

        if self.verbose and fails < max_fail:
            self._debug("========== TRAIN END ==========\n")

        return self.history

    def predict_proba(self, X):
        nn.Module.train(self, False)
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.forward(X_t)
            return torch.sigmoid(logits).cpu().numpy()
