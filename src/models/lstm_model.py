"""
lstm_model.py
-------------
2-layer LSTM model for cholera case forecasting using PyTorch.
Sliding window approach with dropout regularization.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

LSTM_FEATURES = [
    "cases_log", "rainfall_mm", "temperature_c",
    "wash_access_pct", "is_rainy_season", "rainfall_anomaly",
    "week_of_year", "month",
]


class CholeraDataset(Dataset):
    """Sliding window time-series dataset."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.X = torch.FloatTensor(sequences)
        self.y = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMNet(nn.Module):
    """
    2-layer LSTM with dropout and linear output head.

    Architecture:
        Input → LSTM(128, 2 layers, dropout=0.2) → Linear(128→64) → ReLU → Linear(64→1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]   # take last timestep
        return self.fc(last_hidden).squeeze(-1)


class LSTMForecaster:
    """LSTM-based cholera forecaster with PyTorch backend."""

    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "sequence_length": 12,
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "patience": 15,    # early stopping
        }
        self.config = {**default_config, **(config or {})}
        self.model: Optional[LSTMNet] = None
        self.scaler_X: Optional[object] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"LSTM device: {self.device}")

    def _get_features(self, df: pd.DataFrame) -> list:
        return [c for c in LSTM_FEATURES if c in df.columns]

    def _build_sequences(
        self, X: np.ndarray, y: np.ndarray, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (samples, seq_len, features) tensor from time-series arrays."""
        seqs, tgts = [], []
        for i in range(seq_len, len(X)):
            seqs.append(X[i - seq_len:i])
            tgts.append(y[i])
        return np.array(seqs), np.array(tgts)

    def _scale(self, X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Min-max scale features."""
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        self.scaler_X = scaler
        return X_train_s, X_val_s

    def fit(
        self, df: pd.DataFrame, district: Optional[str] = None, test_size: float = 0.2
    ) -> "LSTMForecaster":
        """Train LSTM with early stopping."""
        if district:
            df = df[df["district"] == district].copy()
        df = df.sort_values("date").reset_index(drop=True)

        feat_cols = self._get_features(df)
        X_raw = df[feat_cols].fillna(0).values
        y_raw = df["cases_log"].fillna(0).values if "cases_log" in df.columns else np.log1p(df["cases"].values)

        split = int(len(X_raw) * (1 - test_size))
        X_train_s, X_val_s = self._scale(X_raw[:split], X_raw[split:])

        seq_len = self.config["sequence_length"]
        X_tr_seq, y_tr_seq = self._build_sequences(X_train_s, y_raw[:split], seq_len)
        X_val_seq, y_val_seq = self._build_sequences(X_val_s, y_raw[split:], seq_len)

        if len(X_tr_seq) == 0:
            logger.warning("Insufficient data for LSTM. Skipping fit.")
            return self

        train_ds = CholeraDataset(X_tr_seq, y_tr_seq)
        val_ds = CholeraDataset(X_val_seq, y_val_seq)
        train_loader = DataLoader(train_ds, batch_size=self.config["batch_size"], shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.config["batch_size"], shuffle=False)

        self.model = LSTMNet(
            input_size=len(feat_cols),
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config["epochs"]):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += criterion(self.model(xb), yb).item()

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.config["patience"]:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

        logger.info(f"LSTM fitted | District: {district or 'all'} | Features: {len(feat_cols)}")
        return self

    def predict(self, df: pd.DataFrame, district: Optional[str] = None) -> np.ndarray:
        """Generate predictions on original case scale."""
        if self.model is None:
            logger.warning("LSTM not fitted, returning zeros.")
            return np.zeros(len(df))

        if district:
            df = df[df["district"] == district].copy()
        df = df.sort_values("date").reset_index(drop=True)
        feat_cols = self._get_features(df)
        X_raw = df[feat_cols].fillna(0).values

        if self.scaler_X:
            X_scaled = self.scaler_X.transform(X_raw)
        else:
            X_scaled = X_raw

        seq_len = self.config["sequence_length"]
        X_seq, _ = self._build_sequences(X_scaled, np.zeros(len(X_scaled)), seq_len)

        if len(X_seq) == 0:
            return np.zeros(len(df))

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_seq).to(self.device)
            preds_log = self.model(X_t).cpu().numpy()

        # Pad front with NaN (sequence warm-up)
        full_preds = np.full(len(df), np.nan)
        full_preds[seq_len:] = np.expm1(preds_log).clip(0)
        return full_preds
