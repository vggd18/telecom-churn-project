import torch
import torch.nn as nn
import torch.optim as optim

from base_model import BaseModel


class STabModel(BaseModel, nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        dropout=0.1,
        lr=1e-3,
        name="STabTransformer",
        random_state=42
    ):
        BaseModel.__init__(self, name=name, random_state=random_state)
        nn.Module.__init__(self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cada feature vira um token
        self.feature_embed = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

        self.history = {
            "train_loss": [],
            "val_loss": []
/*******  30c4cf97-e322-4b4a-8d07-7f20d1ccd841  *******/
        }

        self.model = self
        self.to(self.device)
