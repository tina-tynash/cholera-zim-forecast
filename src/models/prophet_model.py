"""
prophet_model.py
----------------
Facebook Prophet model for weekly cholera case forecasting.
Includes: external climate regressors, seasonality tuning, evaluation.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ProphetForecaster:
    """
    Prophet-based cholera forecaster with climate regressors.

    Model specification:
        y(t) = g(t) + s(t) + h(t) + β·X(t) + ε_t

    where:
        g(t) = piecewise-linear trend
        s(t) = Fourier-series seasonality
        h(t) = intervention holidays
        X(t) = [rainfall_mm, temperature_c, wash_access_pct]
    """

    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "seasonality_mode": "multiplicative",
            "yearly_seasonality": True,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "interval_width": 0.95,
        }
        self.config = {**default_config, **(config or {})}
        self.model: Optional[Prophet] = None
        self.district: Optional[str] = None

    def _build_model(self) -> Prophet:
        """Instantiate and configure Prophet model."""
        m = Prophet(
            changepoint_prior_scale=self.config["changepoint_prior_scale"],
            seasonality_prior_scale=self.config["seasonality_prior_scale"],
            seasonality_mode=self.config["seasonality_mode"],
            yearly_seasonality=self.config["yearly_seasonality"],
            weekly_seasonality=self.config["weekly_seasonality"],
            daily_seasonality=self.config["daily_seasonality"],
            interval_width=self.config["interval_width"],
        )
        # Add Zimbabwe rainy-season seasonality
        m.add_seasonality(name="rainy_season", period=365.25 / 2, fourier_order=5)
        # Add climate regressors
        m.add_regressor("rainfall_mm")
        m.add_regressor("temperature_c")
        m.add_regressor("wash_access_pct")
        m.add_regressor("is_rainy_season")
        return m

    def prepare_data(self, df: pd.DataFrame, district: str) -> pd.DataFrame:
        """Filter and format district data for Prophet (requires ds, y columns)."""
        sub = df[df["district"] == district].copy()
        sub = sub.sort_values("date").reset_index(drop=True)

        prophet_df = pd.DataFrame({
            "ds": sub["date"],
            "y": np.log1p(sub["cases"]),   # log-transform for stability
            "rainfall_mm": sub["rainfall_mm"].fillna(0),
            "temperature_c": sub["temperature_c"].fillna(sub["temperature_c"].median()),
            "wash_access_pct": sub["wash_access_pct"].fillna(0.6),
            "is_rainy_season": sub["is_rainy_season"].fillna(0),
        })
        return prophet_df.dropna(subset=["ds", "y"])

    def fit(self, df: pd.DataFrame, district: str) -> "ProphetForecaster":
        """Fit Prophet model to a single district's data."""
        self.district = district
        prophet_df = self.prepare_data(df, district)
        self.model = self._build_model()
        self.model.fit(prophet_df)
        logger.info(f"Prophet fitted | District: {district} | Rows: {len(prophet_df)}")
        return self

    def predict(self, df: pd.DataFrame, horizon_weeks: int = 4) -> pd.DataFrame:
        """
        Generate forecasts for training period + future horizon.

        Returns DataFrame with columns: ds, yhat, yhat_lower, yhat_upper (original scale).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        prophet_df = self.prepare_data(df, self.district)
        future = self.model.make_future_dataframe(periods=horizon_weeks, freq="W")

        # Fill regressors for future dates
        last_row = prophet_df.iloc[-1]
        for col in ["rainfall_mm", "temperature_c", "wash_access_pct", "is_rainy_season"]:
            if col not in future.columns:
                future[col] = last_row[col]
            future[col] = future[col].fillna(last_row[col])

        forecast = self.model.predict(future)

        # Back-transform from log scale
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast[col] = np.expm1(forecast[col]).clip(0)

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend",
                          "yearly", "rainy_season"]]

    def cross_validate(
        self, df: pd.DataFrame, district: str,
        initial: str = "730 days",
        period: str = "90 days",
        horizon: str = "30 days",
    ) -> pd.DataFrame:
        """Run Prophet cross-validation and return metrics."""
        self.fit(df, district)
        prophet_df = self.prepare_data(df, district)
        self.model.fit(prophet_df)
        cv_results = cross_validation(
            self.model, initial=initial, period=period, horizon=horizon, parallel="processes"
        )
        metrics = performance_metrics(cv_results)
        logger.info(f"Cross-validation MAPE: {metrics['mape'].mean():.4f}")
        return metrics
