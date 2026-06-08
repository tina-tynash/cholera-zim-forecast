"""
ensemble.py
-----------
Stacked ensemble combining Prophet, XGBoost, and LSTM predictions.
Meta-learner: Ridge regression with conformal prediction intervals.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.models.prophet_model import ProphetForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.lstm_model import LSTMForecaster

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class EnsembleForecaster:
    """
    Ridge-regression stacked ensemble.

    Level-0: Prophet, XGBoost, LSTM
    Level-1: Ridge meta-learner trained on out-of-fold predictions
    Uncertainty: Conformal prediction intervals (95% coverage)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.prophet = ProphetForecaster(self.config.get("prophet"))
        self.xgb = XGBoostForecaster(self.config.get("xgboost"))
        self.lstm = LSTMForecaster(self.config.get("lstm"))
        self.meta: Optional[Ridge] = None
        self.meta_scaler: Optional[StandardScaler] = None
        self.calibration_residuals: np.ndarray = np.array([])
        self.district: Optional[str] = None

    def _get_xgb_preds(self, df: pd.DataFrame, district: str) -> np.ndarray:
        """Get XGBoost predictions aligned to dataframe index."""
        preds = self.xgb.predict(df, district)
        return np.nan_to_num(preds, nan=0.0)

    def _get_lstm_preds(self, df: pd.DataFrame, district: str) -> np.ndarray:
        """Get LSTM predictions aligned to dataframe index."""
        preds = self.lstm.predict(df, district)
        return np.nan_to_num(preds, nan=0.0)

    def _get_prophet_preds(self, df: pd.DataFrame, district: str) -> np.ndarray:
        """Get Prophet in-sample predictions aligned to dataframe rows."""
        forecast = self.prophet.predict(df, horizon_weeks=0)
        district_df = df[df["district"] == district].sort_values("date")
        merged = district_df[["date"]].merge(
            forecast.rename(columns={"ds": "date", "yhat": "prophet_pred"}),
            on="date", how="left"
        )
        return merged["prophet_pred"].fillna(0).values

    def fit(self, df: pd.DataFrame, district: str) -> "EnsembleForecaster":
        """
        Train all base models and meta-learner.

        Uses time-series cross-validation to generate out-of-fold predictions
        for the meta-learner to avoid data leakage.
        """
        self.district = district
        logger.info(f"Training ensemble | District: {district}")

        district_df = df[df["district"] == district].sort_values("date").reset_index(drop=True)

        # 1. Fit base models on full data
        logger.info("  Fitting Prophet...")
        self.prophet.fit(df, district)

        logger.info("  Fitting XGBoost...")
        self.xgb.fit(df, district, test_size=0.15)

        logger.info("  Fitting LSTM...")
        self.lstm.fit(df, district, test_size=0.15)

        # 2. Generate meta-features via time-series CV
        logger.info("  Generating OOF meta-features...")
        n = len(district_df)
        oof_prophet = np.zeros(n)
        oof_xgb = np.zeros(n)
        oof_lstm = np.zeros(n)

        tscv = TimeSeriesSplit(n_splits=4)
        for train_idx, val_idx in tscv.split(district_df):
            train_sub = df[df["district"] == district].iloc[train_idx]
            val_sub = df[df["district"] == district].iloc[val_idx]

            # XGBoost OOF
            tmp_xgb = XGBoostForecaster(self.config.get("xgboost"))
            tmp_xgb.fit(train_sub.assign(district=district), district, test_size=0.1)
            oof_xgb[val_idx] = tmp_xgb.predict(val_sub.assign(district=district), district)

            # LSTM OOF (only if enough data)
            if len(train_idx) > 30:
                tmp_lstm = LSTMForecaster(self.config.get("lstm"))
                tmp_lstm.fit(train_sub.assign(district=district), district, test_size=0.1)
                preds = tmp_lstm.predict(val_sub.assign(district=district), district)
                oof_lstm[val_idx] = np.nan_to_num(preds, nan=0.0)

            # Prophet OOF
            tmp_prop = ProphetForecaster(self.config.get("prophet"))
            tmp_prop.fit(train_sub.assign(district=district), district)
            forecast = tmp_prop.predict(train_sub.assign(district=district), horizon_weeks=len(val_idx))
            future_preds = forecast.tail(len(val_idx))["yhat"].values
            oof_prophet[val_idx] = future_preds[:len(val_idx)]

        # 3. Fit Ridge meta-learner on OOF predictions
        actual = district_df["cases"].values
        meta_X = np.column_stack([oof_prophet, oof_xgb, oof_lstm])

        # Remove warmup rows (zeros from LSTM)
        valid_mask = (meta_X.sum(axis=1) > 0) & (actual > 0)
        self.meta_scaler = StandardScaler()
        meta_X_scaled = self.meta_scaler.fit_transform(meta_X[valid_mask])
        actual_log = np.log1p(actual[valid_mask])

        self.meta = Ridge(alpha=1.0)
        self.meta.fit(meta_X_scaled, actual_log)

        # 4. Calibrate conformal intervals
        meta_preds_log = self.meta.predict(meta_X_scaled)
        self.calibration_residuals = np.abs(actual_log - meta_preds_log)

        logger.info(
            f"Ensemble fitted | Meta weights: "
            f"Prophet={self.meta.coef_[0]:.3f}, "
            f"XGBoost={self.meta.coef_[1]:.3f}, "
            f"LSTM={self.meta.coef_[2]:.3f}"
        )
        return self

    def predict(
        self, df: pd.DataFrame, district: Optional[str] = None,
        coverage: float = 0.95,
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions with conformal intervals.

        Returns:
            DataFrame with columns: date, district, yhat, yhat_lower, yhat_upper.
        """
        district = district or self.district
        district_df = df[df["district"] == district].sort_values("date").copy()

        # Base model predictions
        prophet_preds = self._get_prophet_preds(df, district)
        xgb_preds = self._get_xgb_preds(df, district)
        lstm_preds = self._get_lstm_preds(df, district)

        meta_X = np.column_stack([prophet_preds, xgb_preds, lstm_preds])

        if self.meta is not None and self.meta_scaler is not None:
            meta_X_scaled = self.meta_scaler.transform(meta_X)
            preds_log = self.meta.predict(meta_X_scaled)
            preds = np.expm1(preds_log).clip(0)

            # Conformal prediction interval
            q = np.quantile(self.calibration_residuals, coverage)
            lower = np.expm1(preds_log - q).clip(0)
            upper = np.expm1(preds_log + q).clip(0)
        else:
            # Fallback: simple average
            preds = (prophet_preds + xgb_preds + lstm_preds) / 3
            std = np.std([prophet_preds, xgb_preds, lstm_preds], axis=0)
            lower = (preds - 1.96 * std).clip(0)
            upper = preds + 1.96 * std

        return pd.DataFrame({
            "date": district_df["date"].values,
            "district": district,
            "yhat": preds.round(1),
            "yhat_lower": lower.round(1),
            "yhat_upper": upper.round(1),
            "prophet_pred": prophet_preds.round(1),
            "xgb_pred": xgb_preds.round(1),
            "lstm_pred": lstm_preds.round(1),
        })

    def save(self, path: Optional[str] = None) -> str:
        """Persist ensemble to disk."""
        path = path or str(MODEL_DIR / f"ensemble_{self.district}.joblib")
        joblib.dump(self, path)
        logger.info(f"Ensemble saved → {path}")
        return path

    @staticmethod
    def load(path: str) -> "EnsembleForecaster":
        """Load ensemble from disk."""
        return joblib.load(path)
