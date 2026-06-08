"""
Feature Engineering for Zimbabwe Cholera Forecasting.

Builds lagged features, rolling statistics, rainfall anomalies,
and WASH interaction terms required by downstream models.

Usage:
    from src.data.feature_engineering import build_features
    df_feat = build_features(panel_df)
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LAG_WEEKS     = [1, 2, 3, 4, 8, 12]
ROLLING_WINS  = [4, 8, 12]
CLIMATE_LAGS  = [1, 2, 3, 4]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based time features."""
    df = df.copy()
    df["week_of_year"]  = df["date"].dt.isocalendar().week.astype(int)
    df["month"]         = df["date"].dt.month
    df["year"]          = df["date"].dt.year
    df["quarter"]       = df["date"].dt.quarter
    df["is_rainy_season"] = df["month"].isin([11, 12, 1, 2, 3]).astype(int)
    # Cyclical encoding for week-of-year
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    return df


def add_case_lags(df: pd.DataFrame, lag_weeks: List[int] = LAG_WEEKS) -> pd.DataFrame:
    """Add lagged case counts and log-transformed lags."""
    df = df.copy()
    df = df.sort_values(["district", "date"])
    for lag in lag_weeks:
        col = f"cases_lag{lag}w"
        df[col] = df.groupby("district")["cases"].shift(lag)
        df[f"log_{col}"] = np.log1p(df[col])
    return df


def add_rolling_features(df: pd.DataFrame, windows: List[int] = ROLLING_WINS) -> pd.DataFrame:
    """Add rolling mean, std, and max for case counts."""
    df = df.copy()
    df = df.sort_values(["district", "date"])
    for w in windows:
        grp = df.groupby("district")["cases"]
        df[f"cases_roll{w}w_mean"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"cases_roll{w}w_std"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        df[f"cases_roll{w}w_max"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
    return df


def add_climate_lags(df: pd.DataFrame, lags: List[int] = CLIMATE_LAGS) -> pd.DataFrame:
    """Add lagged rainfall and temperature features."""
    df = df.copy()
    df = df.sort_values(["district", "date"])
    for lag in lags:
        df[f"rainfall_lag{lag}w"] = df.groupby("district")["rainfall_mm"].shift(lag)
        df[f"temp_lag{lag}w"]     = df.groupby("district")["temperature_c"].shift(lag)
    # Cumulative 4-week rainfall
    df["rainfall_cum4w"] = df.groupby("district")["rainfall_mm"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).sum()
    )
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add WASH × rainfall interaction terms and risk index.

    The interaction captures the amplifying effect of rainfall in low-WASH settings.
    """
    df = df.copy()
    wash_col = "wash_coverage_pct" if "wash_coverage_pct" in df.columns else "wash_coverage"
    rain_col = "rainfall_mm"

    if wash_col in df.columns and rain_col in df.columns:
        wash_inv = 1.0 - df[wash_col].fillna(50) / 100.0
        df["wash_x_rainfall"]  = wash_inv * df[rain_col].fillna(0)
        df["wash_x_anomaly"]   = wash_inv * df.get("rainfall_anomaly_mm", 0).fillna(0)
        df["risk_index"] = (
            0.4 * wash_inv +
            0.3 * (df[rain_col].fillna(0) / 200).clip(0, 1) +
            0.3 * (df.get("poverty_index", pd.Series(0.5, index=df.index)).fillna(0.5))
        )
    return df


def add_growth_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add week-over-week case growth rate (bounded)."""
    df = df.copy()
    df = df.sort_values(["district", "date"])
    prev = df.groupby("district")["cases"].shift(1)
    df["case_growth_rate"] = ((df["cases"] - prev) / (prev + 1)).clip(-5, 5)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Merged panel from ETL pipeline.

    Returns
    -------
    pd.DataFrame
        Panel with all engineered features added.
    """
    logger.info("Building features...")
    df = add_time_features(df)
    df = add_case_lags(df)
    df = add_rolling_features(df)
    df = add_climate_lags(df)
    df = add_interaction_features(df)
    df = add_growth_rate(df)

    n_features = len(df.columns)
    logger.info(f"Features built: {n_features} columns total")
    return df


FEATURE_COLS = [
    "cases_lag1w", "cases_lag2w", "cases_lag3w", "cases_lag4w",
    "cases_lag8w", "cases_lag12w",
    "log_cases_lag1w", "log_cases_lag2w",
    "cases_roll4w_mean", "cases_roll8w_mean", "cases_roll12w_mean",
    "cases_roll4w_std", "cases_roll8w_std",
    "cases_roll4w_max",
    "rainfall_lag1w", "rainfall_lag2w", "rainfall_lag3w", "rainfall_lag4w",
    "rainfall_cum4w", "temp_lag1w",
    "rainfall_anomaly_mm", "rainfall_anomaly_pct",
    "wash_x_rainfall", "wash_x_anomaly", "risk_index",
    "case_growth_rate",
    "week_sin", "week_cos", "is_rainy_season",
    "ocv_active", "wash_active",
    "poverty_index", "population_density_km2",
]
