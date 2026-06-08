"""
xgboost_model.py
----------------
XGBoost gradient-boosted trees for cholera case forecasting.
Includes: feature importance, SHAP values, early stopping.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "cases_lag1w", "cases_lag2w", "cases_lag4w", "cases_lag8w", "cases_lag12w",
    "cases_roll4w_mean", "cases_roll8w_mean", "cases_roll13w_mean",
    "cases_roll4w_std", "cases_roll8w_std",
    "rainfall_mm", "rainfall_lag2w", "rainfall_lag4w",
    "rainfall_anomaly", "rainfall_4w_cumulative",
    "temperature_c", "humidity_pct",
    "wash_access_pct", "wash_rain_interaction",
    "poverty_rate", "pop_density_km2",
    "week_of_year", "month", "year_idx", "is_rainy_season",
    "intervention_count", "ocv_events", "wash_events",
]


class XGBoostForecaster:
    """
    XGBoost forecaster with SHAP interpretability.

    Trained on engineered lag/climate/WASH features.
    Uses time-series cross-validation to prevent data leakage.
    """

    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "early_stopping_rounds": 50,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.config = {**default_config, **(config or {})}
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: List[str] = []
        self.shap_explainer: Optional[shap.TreeExplainer] = None

    def _get_features(self, df: pd.DataFrame) -> List[str]:
        """Return available feature columns from the dataframe."""
        return [c for c in FEATURE_COLS if c in df.columns]

    def _prepare_xy(
        self, df: pd.DataFrame, district: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix X and target y."""
        if district:
            df = df[df["district"] == district].copy()
        df = df.sort_values("date").dropna(subset=["cases"])

        features = self._get_features(df)
        X = df[features].fillna(0)
        y = np.log1p(df["cases"])   # log1p target
        return X, y, df["date"]

    def fit(
        self, df: pd.DataFrame, district: Optional[str] = None,
        test_size: float = 0.2,
    ) -> "XGBoostForecaster":
        """
        Fit XGBoost with early stopping on held-out validation set.

        Args:
            df: Feature-engineered dataframe.
            district: Filter to a single district (or None for all districts).
            test_size: Fraction of data for validation.
        """
        X, y, dates = self._prepare_xy(df, district)
        split = int(len(X) * (1 - test_size))
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        self.feature_names = list(X.columns)
        self.model = xgb.XGBRegressor(
            **{k: v for k, v in self.config.items() if k != "early_stopping_rounds"},
            early_stopping_rounds=self.config["early_stopping_rounds"],
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.shap_explainer = shap.TreeExplainer(self.model)
        best_iter = self.model.best_iteration
        logger.info(
            f"XGBoost fitted | District: {district or 'all'} | "
            f"Best iter: {best_iter} | Features: {len(self.feature_names)}"
        )
        return self

    def predict(self, df: pd.DataFrame, district: Optional[str] = None) -> np.ndarray:
        """
        Predict cholera cases (original scale).

        Returns:
            np.ndarray of predicted case counts.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        X, _, _ = self._prepare_xy(df, district)
        preds_log = self.model.predict(X)
        return np.expm1(preds_log).clip(0)

    def get_shap_values(self, df: pd.DataFrame, district: Optional[str] = None) -> pd.DataFrame:
        """
        Compute SHAP feature importance values.

        Returns:
            DataFrame of SHAP values, one column per feature.
        """
        if self.shap_explainer is None:
            raise RuntimeError("Fit model before computing SHAP values.")
        X, _, _ = self._prepare_xy(df, district)
        shap_values = self.shap_explainer.shap_values(X)
        shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
        return shap_df

    def feature_importance(self) -> pd.DataFrame:
        """Return sorted feature importances (gain)."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        scores = self.model.get_booster().get_score(importance_type="gain")
        fi = (
            pd.DataFrame.from_dict(scores, orient="index", columns=["importance"])
            .sort_values("importance", ascending=False)
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        return fi

    def cross_validate_ts(
        self, df: pd.DataFrame, district: Optional[str] = None, n_splits: int = 5
    ) -> Dict:
        """Time-series cross-validation with rolling origin."""
        X, y, _ = self._prepare_xy(df, district)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mapes, rmses = [], []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            m = xgb.XGBRegressor(
                **{k: v for k, v in self.config.items() if k != "early_stopping_rounds"},
            )
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = np.expm1(m.predict(X_val)).clip(0)
            actual = np.expm1(y_val.values).clip(0)

            mape = np.mean(np.abs(actual - preds) / (actual + 1)) * 100
            rmse = np.sqrt(np.mean((actual - preds) ** 2))
            mapes.append(mape)
            rmses.append(rmse)

        results = {
            "mean_mape": np.mean(mapes),
            "std_mape": np.std(mapes),
            "mean_rmse": np.mean(rmses),
            "std_rmse": np.std(rmses),
            "fold_mapes": mapes,
        }
        logger.info(f"CV MAPE: {results['mean_mape']:.2f}% ± {results['std_mape']:.2f}%")
        return results
